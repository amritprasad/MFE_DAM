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
'''
Sample Strategy on full market
'''
retMonth=((1+us_df).groupby(pd.Grouper(freq='M'))).prod()-1
retMonth.head()

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