"""
MFE 230K - Factor Timing Project

Authors: Ms. Allocation
"""

# Imports
import pandas as pd

# Import functions
import functions as fnc

# Set options
pd.set_option('display.max_columns', 20)
# %%
# Read in data for US
# us_df = fnc.load_data('US')
us_df = pd.read_csv('./Data/US.csv', parse_dates=[0], index_col=[0])
# Look at rolling correlations of factors vs the other
fnc.rolling_corr(us_df, corrwith='MKT', window=126, title='US_MKT')  # ICs
fnc.rolling_corr(us_df, corrwith='VAL', window=126, title='US_VAL')
fnc.rolling_corr(us_df, corrwith='MOM', window=126, title='US_MOM')
fnc.rolling_corr(us_df, corrwith='QUAL', window=126, title='US_QUAL')
# %%
# Calculate IRs
# IRs wrt Market
ir_mkt_df, period_start, period_end = fnc.calc_ir(us_df, bench='MKT', freq='M')
# IRs wrt Risk-Free (basically Sharpe)
ir_rf_df, _, _ = fnc.calc_ir(us_df, bench='RF', freq='M')

# The IRs wrt the market are very low for the factors implying that their
# residual returns are usually not very significant. The IR for the risk-free
# vs the Mkt is upward-biased because it doesn't change much during the year
# causing its residual risk (std deviation) to be very low. It shouldn't be
# taken as indicative of great performance.

# Check persistence of the IRs (only freq='Y' makes sense)
print(ir_mkt_df.apply(lambda x: x.autocorr(), axis=0))
print(ir_rf_df.apply(lambda x: x.autocorr(), axis=0))
# %%
# Predict IRs
# Get features for IR prediction
# garch_df = fnc.vol_GARCH(mkt_ret, period_start, period_end)
garch_df = pd.read_csv('./Data/US_GARCH_1m.csv', parse_dates=[0],
                       index_col=[0])

# %%
# Implement the CAPE valuation timing strategy
# Equal static weights and neutral weights of 0.5
w = fnc.cape_timing(rolling_window=60*12, neutral_wt=0.5, freq='M',
                    static_wts=[1/3]*3, fac_names=['VAL', 'MOM', 'QUAL'])
macro_df = fnc.macro_data('US', 12)
