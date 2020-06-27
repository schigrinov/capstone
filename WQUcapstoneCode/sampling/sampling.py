import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm
import scipy.stats as stats

# import visual tools
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def sampled_bars(df, sampling_column, m):
    t = df[sampling_column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += x
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx

def sampled_bar_df(df, sampling_column, m):
    idx = sampled_bars(df, sampling_column, m)
    return df.iloc[idx].drop_duplicates()


def select_sample_data(ref, sub, price_col, start_date, end_date):
    '''
    select a sample of data based on date, assumes datetimeindex

    # args
        ref: pd.DataFrame containing all ticks
        sub: subordinated pd.DataFrame of prices
        price_col: str(), price column
        date: str(), date to select
    # returns
        xdf: ref pd.Series
        xtdf: subordinated pd.Series
    '''
    mask = (ref.index > start_date) & (ref.index <= end_date)
    xdf = ref[price_col][mask]
    mask = (sub.index > start_date) & (sub.index <= end_date)
    xtdf = sub[price_col][mask]
    return xdf, xtdf


def plot_sample_data(ref, sub, bar_type, *args, **kwds):
    f, axes = plt.subplots(3, sharex=True, sharey=True, figsize=(10, 7))
    ref.plot(*args, **kwds, ax=axes[0], label='price')
    sub.plot(*args, **kwds, ax=axes[0], marker='X', ls='', label=bar_type)
    axes[0].legend();

    ref.plot(*args, **kwds, ax=axes[1], label='price', marker='o')
    sub.plot(*args, **kwds, ax=axes[2], ls='', marker='X',
             color='r', label=bar_type)

    for ax in axes[1:]: ax.legend()
    plt.tight_layout()


def returns(s):
    '''Log returns of a data series'''
    arr = np.diff(np.log(s))
    return pd.Series(arr, index=s.index[1:])


def get_test_stats(bar_types,bar_returns,test_func,*args,**kwds):
    '''
    Wrapper to display the data nicely
    :param bar_types: labels
    :param bar_returns: list of return series
    :param test_func: function to use for this stat
    :param args: args for a test function
    :param kwds:
    :return: labeled dataFrame with stats
    '''
    dct = {bar:(int(bar_ret.shape[0]), test_func(bar_ret,*args,**kwds))
           for bar,bar_ret in zip(bar_types,bar_returns)}
    df = (pd.DataFrame.from_dict(dct)
          .rename(index={0:'sample_size',1:f'{test_func.__name__}_stat'})
          .T)
    return df


def plot_autocorr(bar_types,bar_returns, ylim=[-0.2,0.2]):
    f,axes=plt.subplots(len(bar_types),figsize=(10,7))
    for i, (bar, typ) in enumerate(zip(bar_returns, bar_types)):
        axes[i].acorr(bar, usevlines=True,
                  maxlags=50, normed=True, lw=2)
        axes[i].set_ylabel(typ)
        axes[i].set_ylim(ylim)
    plt.tight_layout()


def plot_hist(bar_types,bar_returns):
    f,axes=plt.subplots(len(bar_types),figsize=(10,6))
    for i, (bar, typ) in enumerate(zip(bar_returns, bar_types)):
        g = sns.distplot(bar, ax=axes[i], kde=False, label=typ)
        g.set(yscale='log')
        axes[i].legend()
    plt.tight_layout()

def jb(x,test=True):
    '''Jarque bera test (smaller - closer to normal) '''
    np.random.seed(1)
    if test: return stats.jarque_bera(x)[0]
    return stats.jarque_bera(x)[1]

def shapiro(x,test=True):
    '''Shapiro-Wilk Test (larger - closer to normal)'''
    np.random.seed(12345678)
    if test: return stats.shapiro(x)[0]
    return stats.shapiro(x)[1]

if __name__ == '__main__':
    pass