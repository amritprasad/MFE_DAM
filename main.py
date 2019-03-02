"""
MFE 230K - Factor Timing Project

Authors: Ms. Allocation
"""

# Imports
import pandas as pd

# Import functions
import functions as fnc
# %%
# Read in data
us_df = fnc.load_data('US')
# Look at rolling correlations of factors vs the other
fnc.rolling_corr(us_df, corrwith='MKT', window=126, title='US_MKT')
fnc.rolling_corr(us_df, corrwith='VAL', window=126, title='US_VAL')
fnc.rolling_corr(us_df, corrwith='MOM', window=126, title='US_MOM')
fnc.rolling_corr(us_df, corrwith='QUAL', window=126, title='US_QUAL')
