import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm
import pyfolio as pf

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._split import _BaseKFold
from sklearn.metrics import *


def get_feature_types(X_train, Y_train, \
                      rfc = RandomForestClassifier(criterion='entropy', class_weight='balanced_subsample', \
                                                   bootstrap=False, random_state=1)): # these are RF parameters recommended in the book)
    all_feature_cols = list(X_train.columns)
    frdif_feature_dict = {c[:-6]: c for c in all_feature_cols if c[-5:] == 'frdif'}
    ex_frdiff_cols = [c for c in all_feature_cols if c not in frdif_feature_dict.values()]
    frdiff_cols = [c for c in all_feature_cols if c not in frdif_feature_dict.keys()]

    rfc.fit(X_train,Y_train)
    # selecting the most important features
    feat_imp = pd.DataFrame({'Importance':rfc.feature_importances_})
    feat_imp['feature'] = X_train.columns
    top_feat = list(feat_imp[feat_imp.Importance>0.01].feature)

    return all_feature_cols, ex_frdiff_cols, frdiff_cols, top_feat


def get_pyfolio_simple_tear_sheet(mdl,X_tr, Y_tr, X_tst, Y_tst, rtns_actual):
    # 1. Fit model using training set
    mdl.fit(X_tr, Y_tr)
    # 2. Predict labels using trained model
    predicted_labels = pd.Series(mdl.predict(X_tst), index = Y_tst.index)
    # 3. Calculate return using the predicted labels together the return vector
    rtns = predicted_labels * rtns_actual
    # 4. Generate term sheet using the backtest portfolio return vector using PyFolio package
    pf.create_simple_tear_sheet(rtns)


def train_valid_test_split(data, proportions='50:25:25'):
    """
    Splits the data into 3 parts - training, validation and test sets
    :param proportions: proportions for the split, like 2:1:1 or 50:30:20
    :param data: preprocessed data
    :return: X_train, Y_train, target_rtns_train, X_valid, Y_valid, target_rtns_valid, X_test, Y_test, target_rtns_test
    """
    features = [c for c in data.columns if c not in ('ret','bin')]
    n = len(data)
    borders = [float(p) for p in proportions.split(':')]
    borders = borders / np.sum(borders)

    train_ids = (0, int(np.floor(n * borders[0])))
    valid_ids = (train_ids[1] + 1, int(np.floor(n * np.sum(borders[:2]))))
    test_ids = (valid_ids[1] + 1, n)

    X_train = data[features].iloc[train_ids[0]:train_ids[1], :]
    X_valid = data[features].iloc[valid_ids[0]:valid_ids[1], :]
    X_test = data[features].iloc[test_ids[0]:test_ids[1], :]

    Y_train = data.bin.iloc[train_ids[0]:train_ids[1]]
    Y_valid = data.bin.iloc[valid_ids[0]:valid_ids[1]]
    Y_test = data.bin.iloc[test_ids[0]:test_ids[1]]

    target_rtns_train = data.ret.iloc[train_ids[0]:train_ids[1]]
    target_rtns_valid = data.ret.iloc[valid_ids[0]:valid_ids[1]]
    target_rtns_test = data.ret.iloc[test_ids[0]:test_ids[1]]

    return X_train, Y_train, target_rtns_train, X_valid, Y_valid, target_rtns_valid, X_test, Y_test, target_rtns_test



def cv_split(size, n_splits=3):
    indices = np.arange(size)
    test_starts = [
        (i[0], i[-1] + 1) for i in np.array_split(np.arange(size), n_splits)
    ]

    for i, j in test_starts:
        test_indices = indices[i:j]
        train_indices = np.array(list(set(indices) - set(test_indices)))
        yield  train_indices, test_indices


# TODO: can be easily improved using numba
def cv_with_custom_score(mdl, X, Y, rtn, n_folds=3):
    '''
    Calculates average annualised daily return and sharp ratio based pn cross-validation
    (result may be different from pyfolio, but good enough for our purpose)
    :param mdl: model
    :param X: features
    :param Y: labels
    :param rtn: returns
    :param n_folds: splits in cross validation
    :return: avg. daily return and sharp ratio (annualised)
    '''
    cv = cv_split(Y.shape[0], n_folds)
    rtns_testing = pd.Series()
    for split in cv:
        X_train, Y_train = X.iloc[split[0]], Y.iloc[split[0]]
        X_test = X.iloc[split[1]]
        case_rtn = mdl.fit(X_train, Y_train).predict(X_test) * rtn.iloc[split[1]]
        rtns_testing = rtns_testing.append(case_rtn)
    
    # Calculate the number of days between as we are using intraday data with 4-hour interval rather than daily data.
    days = np.busday_count(Y.index.min().date(),Y.index.max().date())
    
    # Here we annualize daily volatility using the squaring of time rule, by assuming the daily returns are iid.    
    rtns_volatility_test = np.std(rtns_testing.groupby(rtns_testing.index.date).sum()) * np.sqrt(252)
    # Here w annualize daily return by using arithmetic return as prooxy
    rtns_testing = np.sum(rtns_testing) * 252 / days 
    
    # Annualize geometric return not implemented
    #(np.cumprod(rtns_testing+1)[-1]-1) * 252 / days

    # function return annualized return and Sharpe Ratio as annualized return - annualized volatility
    return rtns_testing, rtns_testing/rtns_volatility_test


class PurgedKFold(_BaseKFold):
    """
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed continuous (shuffle=False), w/o training samples in between
    """
    def __init__(self,n_splits=3,t1=None,pctEmbargo=0.):
        if not isinstance(t1,pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold,self).__init__(n_splits,shuffle=False,random_state=None)
        self.t1=t1
        self.pctEmbargo=pctEmbargo
        
    def split(self,X,y=None,groups=None):
        if (X.index==self.t1.index).sum()!=len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indices=np.arange(X.shape[0])
        mbrg=int(X.shape[0]*self.pctEmbargo)
        test_starts=[
            (i[0],i[-1]+1) for i in np.array_split(np.arange(X.shape[0]),
                                                   self.n_splits)
        ]
        for i,j in test_starts:
            t0=self.t1.index[i] # start of test set
            test_indices=indices[i:j]
            maxT1Idx=self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices=self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
            if maxT1Idx<X.shape[0]: # right train ( with embargo)
                train_indices=np.concatenate((train_indices, indices[maxT1Idx+mbrg:]))
            yield train_indices,test_indices

from sklearn.model_selection import StratifiedKFold


def crossValPlot(skf,classifier,X_,y_):
    
    X = np.asarray(X_)
    y = np.asarray(y_)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    f,ax = plt.subplots(figsize=(10,7))
    i = 0
    for train, test in skf.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(bbox_to_anchor=(1,1))


def classifier_metrics(X,Y,c, confusion = True):
    y_pred = c.predict_proba(X)[:, 1]
    y_pred_ = c.predict(X)

    if confusion: cm_rf = confusion_matrix(Y, y_pred_)
    fpr_rf, tpr_rf, _ = roc_curve(Y, y_pred)
    print(classification_report(Y, y_pred_))

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    #plt.legend(loc='best')
    if confusion: 
        plt.matshow(cm_rf)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__':
    pass
