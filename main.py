# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:39:29 2019

@author: Raman Gumber
"""

"""
MFE 230K - Factor Timing Project

Authors: Ms. Allocation
"""

# Imports
import pandas as pd
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Import functions
import functions as fnc
import MOMIRPred
import strategyBuilder as strat_build
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
# Set options
pd.set_option('display.max_columns', 20)
# %%
# Read in data for US
# us_df = fnc.load_data('US')
us_df = pd.read_csv('./Data/US.csv', parse_dates=[0], index_col=[0])
# Look at rolling correlations of factors vs the other
US_MKTCorrs=fnc.rolling_corr(us_df, corrwith='MKT', window=126, title='US_MKT')  # ICs
US_VALCorrs=fnc.rolling_corr(us_df, corrwith='VAL', window=126, title='US_VAL')
US_MOMCorrs=fnc.rolling_corr(us_df, corrwith='MOM', window=126, title='US_MOM')
US_QUALCorrs=fnc.rolling_corr(us_df, corrwith='QUAL', window=126, title='US_QUAL')
#%%
def dateConverter(d):
    
    if round(d%1,2)==.1:
        return str(d)+'0'
    else:
        return str(d)
    
shillerEP=pd.read_csv('shillerEP.csv')
shillerEP.Date=[datetime.datetime.strptime(dateConverter(d), "%Y.%m").date() for d in shillerEP.Date]
#shillerEP['PE']=shillerEP['RealPrice']/shillerEP['RealEarnings']
shillerEP['EP']=1/shillerEP['CAPE']
shillerEP['EP95Pct']=shillerEP['EP'].expanding(20).quantile(.95)
shillerEP['EPmedian']=shillerEP['EP'].expanding(20).quantile(.5)
shillerEP['EP5Pct']=shillerEP['EP'].expanding(20).quantile(.05)
shillerEP['ContraWeight']=1+(shillerEP['EP']-shillerEP['EPmedian'])/(shillerEP['EP95Pct']-shillerEP['EP5Pct'])
shillerEP.loc[shillerEP['ContraWeight']<.5,'ContraWeight']=.5
shillerEP.loc[shillerEP['ContraWeight']>1.5,'ContraWeight']=1
shillerEP=shillerEP.set_index('Date')
# %%
# Calculate IRs
# IRs wrt Market
ir_mkt_df, period_start, period_end = fnc.calc_ir(us_df, bench='MKT')
# IRs wrt Risk-Free (basically Sharpe)
ir_rf_df, _, _ = fnc.calc_ir(us_df, bench='RF')

# The IRs wrt the market are very low for the factors implying that their
# residual returns are usually not very significant. The IR for the risk-free
# vs the Mkt is upward-biased because it doesn't change much during the year
# causing its residual risk (std deviation) to be very low. It shouldn't be
# taken as indicative of great performance.

# Check persistence of the IRs
print(ir_mkt_df.apply(lambda x: x.autocorr(), axis=0))
print(ir_rf_df.apply(lambda x: x.autocorr(), axis=0))
# %%
# Predict IRs
# Get features for IR prediction
# garch_df = fnc.vol_GARCH(mkt_ret, period_start, period_end)
garch_df = pd.read_csv('./Data/US_GARCH_1m.csv', parse_dates=[0],
                       index_col=[0])
#%%
retMonthWithContra['CAPE']
pd.Series(pd.cut(bins=[0,10.7,13.7,17.8,22,100],x=retMonthWithContra['CAPE'] ,labels=False, retbins=True, right=False))[0]
#%%
'''
Sample Strategy on full market
'''
retMonth=((1+us_df).groupby(pd.Grouper(freq='M'))).prod()-1
retMonthWithContra=(((1+us_df).groupby(pd.Grouper(freq='MS'))).prod()-1)
retMonthWithContra.index=[d+relativedelta(months=1) for d in retMonthWithContra.index]
retMonthWithContra=retMonthWithContra.join(shillerEP[['ContraWeight','EP','CAPE']])
retMonthWithContra['RContra']=retMonthWithContra['ContraWeight']*retMonthWithContra['MKT']+((1-retMonthWithContra['ContraWeight'])*retMonthWithContra['RF'])
retMonthWithContra['excessMOM']=retMonthWithContra['MOM']-retMonthWithContra['RF']
retMonthWithContra['excessMKT']=retMonthWithContra['MKT']-retMonthWithContra['RF']
retMonthWithContra['excessMKT']=retMonthWithContra['excessMKT'].rolling(12).sum()
retMonthWithContra['excessMOM']=retMonthWithContra['excessMOM'].rolling(12).sum()
(1+retMonthWithContra[['RContra','MKT']]).cumprod().plot()
plt.show()
plt.close()
pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]//.2
#retMonthWithContra['CAPERank']=retMonthWithContra['CAPE'].rolling(60).apply(pctrank)
retMonthWithContra['CAPERank']=pd.Series(pd.cut(bins=[0,10.7,13.7,17.8,22,100],x=retMonthWithContra['CAPE'] ,labels=False, retbins=True, right=False))[0]

def rev_roll(x,lookAhead=120):
    return x.iloc[::-1].rolling(lookAhead, min_periods=0).sum().iloc[::-1]

retMonthWithContra['lookFwdRets']=rev_roll(retMonthWithContra['excessMKT'],120)
retMonthWithContra.iloc[:-120][['MKT','CAPERank']].groupby('CAPERank').mean()
lookAheadRets=pd.concat([pd.concat([retMonthWithContra['CAPERank'],rev_roll(retMonthWithContra['excessMKT'],120)],axis=1).iloc[:-120][['excessMKT','CAPERank']].groupby('CAPERank').mean()/120,\
pd.concat([retMonthWithContra['CAPERank'],rev_roll(retMonthWithContra['excessMKT'],12)],axis=1).iloc[:-12][['excessMKT','CAPERank']].groupby('CAPERank').mean()/12,
pd.concat([retMonthWithContra['CAPERank'],rev_roll(retMonthWithContra['excessMKT'],3)],axis=1).iloc[:-3][['excessMKT','CAPERank']].groupby('CAPERank').mean()/3],axis=1)

lookAheadRets.columns=['10YrLookAhead','1YrLookAhead','3MonthLookAhead']
lookAheadRets.plot(kind='bar',title='Look Ahead Returns by CAPE Rank')
plt.show()
plt.close()
retMonth.head()
retMonthWithContra=retMonthWithContra.dropna()
mod=sm.OLS(retMonthWithContra['RContra']-retMonthWithContra['RF'], sm.add_constant(retMonthWithContra[['excessMOM','excessMKT']]))
res = mod.fit()
#%%

def modelMaker(features):
    allModels={}
    for i in range(len(features.index)):
        try:    
            df=features.loc[features.index[i-120]:features.index[i]]
            mod = sm.OLS(df['RContra']-df['RF'], sm.add_constant(df[['excessMOM','excessMKT']]))
            res = mod.fit()
            allModels[features.index[i+1]]=res
        except:
            pass
    return allModels
def pred(row,allModels):
    try:
     
        val=allModels[row['Date']].predict(1,[row['excessMOM'],row['excessMKT']])[0]
        return np.sign(val)*min(.02,np.abs(val))
    except:
     
        return 0
def predictions(features,allModels,n):
    df=pd.concat([features['MOMBeta'],features['MKT'].rolling(n).sum().shift(n)],axis=1)
    df['Date']=df.index
    df=df.join(features['MOM'])
    df['predMOM']=df.apply(lambda row: pred(row,allModels),axis=1)
    return df[['MOM','predMOM']]
#%%
allModels=modelMaker(retMonthWithContra)
allModelParams={k:allModels[k].params for k in allModels.keys()}
paramDF=pd.DataFrame.from_dict(allModelParams).T
paramDF.resample('10Y').first().plot(title='coeffecients')
#%%
allModels[datetime.datetime(2010,1,1)].summary()
#%%
strat=strat_build.strategies(np.array([1,0,0,0]),retMonth,'RF',datetime.datetime(2010, 1, 1),datetime.datetime(2019, 1, 30))      
rets=strat.get_factor_return_subset()
print(rets.head())
strat.get_plots()

## Annualized Sharpe ratio
print('Sharpe Ratio of Market ',strat.get_sharpe()/np.sqrt(12))
#%%
'''
Build MOM Future predicted IRs
'''

#%%
us_df['MOMBeta']=(us_df['MOM'].rolling(250*10).cov(us_df['MKT']))/(us_df['MOM'].rolling(250*10).var())
features=(retMonth.join(garch_df).join(us_df['MKT'].rolling(90).std().rename('rollingRealizedVol').groupby(pd.Grouper(freq='M')).last())\
          .join(us_df['MOM'].rolling(30).corr(us_df['MKT']\
               .rolling(250).std()).rename('volMomCorr').groupby(pd.Grouper(freq='M')).last()).join(us_df['MOMBeta'].groupby(pd.Grouper(freq='M')).last()))
features['QUALCrash']=features['QUAL']<=features['QUAL'].rolling(120).quantile(.25).shift(1)
features['RVolCrash']=features['rollingRealizedVol']>=features['rollingRealizedVol'].rolling(120).quantile(.75).shift(1)
features.tail(200)
features['RVolCrash'].value_counts()
#%%
'''
Try to optimize for the best lag... seems we might be fine with 36 lags on the lookforward of MOM IR on current MOMBeta and
n period lookback cumulative return of market
'''
plt.plot(list(map(lambda x: MOMIRPred.regModel(x,features)[0].rsquared,np.arange(1,120))))
plt.xlabel('number of lags')
plt.ylabel('OOS Rsqrd')
plt.plot()
plt.show()
plt.close()
#%%
'''
Make a walk forward regression model for every month possible... keep retraining model every month for next month's predictions
'''
n=36
allModels=MOMIRPred.modelMaker(n,features)
#%%
MOMPredDf=MOMIRPred.predictions(features,allModels,n)
MOMIRPred.plotPredvsReal(MOMPredDf)