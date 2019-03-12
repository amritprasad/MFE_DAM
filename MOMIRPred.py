# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:42:01 2019

@author: Raman Gumber
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
def regModel(n,features):
    '''
    Builds a regression model to predict n period lagged fwd MOM IR based on MKT return and MOM's Beta on Marke
    
    args:
        n: number lags to check
        features: dataframe of features that needs to contain MOM,MKT and MOM Beta features
    *** The code does the lagging itself ***
    '''
    n=int(n)
    df=pd.concat([features['MOMBeta'],features['MKT'].rolling(n).sum().shift(n),\
                  features['MOM'].rolling(n).sum().shift(-n)],axis=1).dropna()
    mod = sm.OLS(df['MOM'], sm.add_constant(df[['MOMBeta','MKT']]))
    res = mod.fit()
    return res,df

def modelMaker(n,features):
    allModels={}
    for i in range(len(features.index)):
        try:    
            feats=features.loc[:features.index[i]]
            df=pd.concat([feats['MOMBeta'],feats['MKT'].rolling(n).sum().shift(n),feats['MOM'].rolling(n).sum().shift(-n)],axis=1).dropna()
            df
            mod = sm.OLS(df['MOM'], (df[['MOMBeta','MKT']]))
            res = mod.fit()
            allModels[features.index[i-n+1]]=res
        except:
            pass
    return allModels
def pred(row,allModels):
    try:
     
        val=allModels[row['Date']].predict([row['MOMBeta'],row['MKT']])[0]
        return np.sign(val)*min(.02,np.abs(val))
    except:
     
        return 0
def predictions(features,allModels,n):
    df=pd.concat([features['MOMBeta'],features['MKT'].rolling(n).sum().shift(n)],axis=1)
    df['Date']=df.index
    df=df.join(features['MOM'])
    df['predMOM']=df.apply(lambda row: pred(row,allModels),axis=1)
    return df[['MOM','predMOM']]
def plotPredvsReal(df):

    
    fig, ax = plt.subplots(2,2,figsize=(12,7),sharex=False)
    ax1,ax2,ax3,ax4 = ax.flatten()
    
    ax1.axis('off')
    
    #ax1.set_xticks(features['MOMBeta'].loc[timeStart:timeEnd].index)
    ax1.legend(loc=3)
    
    timeStart=datetime.datetime(1999, 1, 1)
    timeEnd=datetime.datetime(2002, 1, 1)
    (1+df[['MOM','predMOM']].loc[timeStart:timeEnd]).cumprod().plot(ax=ax2)
    
    ax2.legend(loc=3)
    
    
    
    timeStart=datetime.datetime(2006, 1, 1)
    timeEnd=datetime.datetime(2010, 1, 1)
    (1+df[['MOM','predMOM']].loc[timeStart:timeEnd]).cumprod().plot(ax=ax3)
    ax3.legend(loc=3)
    
    
    
    timeStart=datetime.datetime(1999, 1, 1)
    timeEnd=datetime.datetime(2019, 1, 1)
    (1+df[['MOM','predMOM']].loc[timeStart:timeEnd]).cumprod().plot(ax=ax4)
    ax4.legend(loc=3)
    
    #ax4.axis('off')
    
    plt.setp(ax1.get_xticklabels(),visible=True)
    plt.setp(ax2.get_xticklabels(),visible=True)
    plt.setp(ax3.get_xticklabels(),visible=True)
    plt.legend()
    plt.subplots_adjust(wspace=.3)
    plt.plot()
    plt.show()
    plt.close()
