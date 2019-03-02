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
us_df = pd.read_csv('./Data/US.csv')
# Look at rolling correlations of factors vs the other
fnc.rolling_corr(us_df, corrwith='MKT', window=126, title='US_MKT')  # ICs
fnc.rolling_corr(us_df, corrwith='VAL', window=126, title='US_VAL')
fnc.rolling_corr(us_df, corrwith='MOM', window=126, title='US_MOM')
fnc.rolling_corr(us_df, corrwith='QUAL', window=126, title='US_QUAL')
# %%
# Calculate IRs
# IRs wrt Market
ir_mkt_df = fnc.calc_ir(us_df, bench='MKT')
# IRs wrt Risk-Free (basically Sharpe)
ir_rf_df = fnc.calc_ir(us_df, bench='RF')
# The IRs wrt the market are very low for the factors implying that their
# residual returns are usually not very significant. The IR for the risk-free
# vs the Mkt is upward-biased because it doesn't change much during the year
# causing its residual risk (std deviation) to be very low. It shouldn't be
# taken as indicative of great performance.
