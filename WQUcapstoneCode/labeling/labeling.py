import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm

from ..utils.parralel import *


def getDailyVol(close, span0=100, days=1):
    # daily vol reindexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=days))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0.dropna()


def getTEvents_mid(gRaw, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = np.log(gRaw).diff().dropna()
    for i in tqdm(diff.index[1:]):
        pos, neg = float(sPos+diff.loc[i]), float(sNeg+diff.loc[i])
        sPos, sNeg=max(0., pos), min(0., neg)
        if sNeg<-h: sNeg=0;tEvents.append(i)
        elif sPos>h: sPos=0;tEvents.append(i)
    return pd.DatetimeIndex(tEvents)

def getTEvents(gRaw, h):
    if ('ask' not in gRaw) and ('bid' not in gRaw):
        return getTEvents_mid(gRaw, h)
    tEvents, sPos, sNeg = [], 0, 0
    diff = np.log(gRaw).diff()  # bid vs bid and ask vs ask
    diff_short = np.log(gRaw.ask / gRaw.bid.shift(1))  # returns from selling @bid(T-1) and buying @ask(T+0)
    diff_long = np.log(gRaw.bid / gRaw.ask.shift(1))  # returns from buying @ask(T-1) and selling @bid(T+0)
    for i in tqdm(diff.index[1:]):
        pos, neg = sPos + diff_long.loc[i], sNeg + diff_short[i]
        sPos, sNeg = max(0., sPos + diff.ask.loc[i]), min(0., sNeg + diff.bid.loc[i])
        if pos > h:
            sPos = 0;
            tEvents.append(i);
        elif neg < -h:
            sNeg = 0;
            tEvents.append(i)

    return pd.DatetimeIndex(tEvents)


def addVerticalBarrier(tEvents, close, numDays=1):
    """ Generates timeindex of events where the vertical barrier was reached within numDays
    :param tEvents: events when upper or lower barrier was reached
    :param close: dataframe/series of closing prices
    :param numDays: max number of days to hold the position
    :return: sorted pillars
    """
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]] #removing times that are beyond those in consideration
    t1 = (pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]))
    return t1


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    # 1) get target
    # tEvents = [event for event in tEvents if event in trgt.index]
    trgt = trgt[tEvents]  # get target volatility
    trgt = trgt[trgt > minRet]  # filter out returns lower than the minRet threshold
    # 2) get t1 (max holding period)
    if t1 is False: t1 = pd.Series(pd.NaT, index=tEvents)
    # 3) form events object, apply stop loss on t1
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side[trgt.index], ptSl[:2]
    events = (pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1)
              .dropna(subset=['trgt']))
    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index),
                      numThreads=numThreads, close=close, events=events,
                      ptSl=ptSl_)
    events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan
    if side is None: events = events.drop('side', axis=1)
    return events


def applyPtSlOnT1(close, events, ptSl, molecule):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1]  # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # path returns
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # earliest stop loss
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # earliest profit taking
    return out


def getBins(events, close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    # 1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    if 'side' in events_: out['ret'] *= events_['side']  # meta-labeling
    out['bin'] = np.sign(out['ret'])
    out.loc[np.abs(out.ret) < events.trgt[out.index], 'bin'] = 0.
    if 'side' in events_: out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling
    return out


def dropLabels(events, minPct=.05):
    # apply weights, drop labels with insufficient examples
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0] < 3: break
        print('dropped label: ', df0.argmin(), df0.min())
        events = events[events['bin'] != df0.argmin()]
    return events


if __name__ == '__main__':
    pass
