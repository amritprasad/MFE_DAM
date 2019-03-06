# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 23:04:15 2019

@author: Raman Gumber
"""

import datetime
import numpy as np
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pyfolio as pf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class strategies:
    def __init__(self, weights,factor_returns,benchmark,beg_date,end_date):
        self.weights=weights
        self.factor_returns=factor_returns
        self.beg_date=beg_date
        self.end_date=end_date
        self.factors=['MKT', 'VAL', 'MOM', 'QUAL']
        self.factor_return_subset=factor_returns.loc[beg_date:end_date,self.factors]
        self.returns=np.matmul(self.factor_return_subset,weights)
        self.benchmark=factor_returns[benchmark].loc[beg_date:end_date]
        self.factor_return_subset['strategy_return']=self.returns
      
    def get_factor_return_subset(self):
        factor_return_subset=self.factor_return_subset
        return factor_return_subset
    def get_sharpe(self):
        factor_return_subset=self.factor_return_subset
        return pf.timeseries.sharpe_ratio(factor_return_subset['strategy_return'])
    
    def get_plots(self):
        factor_return_subset=self.factor_return_subset
        fig,axes=plt.subplots(2,1,figsize=(10,8))
        ax=axes.flatten()

        pf.plotting.plot_drawdown_periods(factor_return_subset['strategy_return'],ax=ax[0])
        pf.plotting.plot_returns(factor_return_subset['strategy_return'],ax=ax[1])
        
        plt.show()
        plt.close()
    def get_returns(self):
        return self.returns