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
# garch_df = fnc.vol_GARCH(us_df['MKT'].copy(), period_start)
# garch_df = pd.read_csv('./Data/US_GARCH.csv', parse_dates=[0], index_col=[0])
garch_df = pd.read_csv('./Data/vol_garch_US_MKT.csv', parse_dates=[0],
                       index_col=[0], header=None)
garch_df.columns = ['GARCH_1M']
garch_df.index.names = ['DATE']

# %%
# Implement the CAPE valuation timing strategy
# Equal static weights and neutral weights of 0.5
w = fnc.cape_timing(rolling_window=60*12, neutral_wt=0.5, freq='M',
                    static_wts=[1/3]*3, fac_names=['VAL', 'MOM', 'QUAL'])
# %%
# Implement macro factors strategy
# macro_df = fnc.macro_data('US', 12)
macro_df = pd.read_excel('Data/US_Macro_Factors.xlsx', sheet_name='Data',
                         index_col=[0], parse_dates=[0])
state_df = fnc.macro_states(macro_df, style='naive', roll_window=60)
forecast_state_df = fnc.forecast_states(state_df, style='constant')
static_exposure = pd.DataFrame(index=['MKT', 'VAL', 'MOM', 'QUAL'],
                               columns=['Growth', 'Inflation', 'Liquidity',
                                        'Volatility'],
                               data=[[1, 1, -1, 1], [1, -1, -1, -1],
                                     [1, 0, -1, 0], [-1, 0, -1, 1]])

# Try with shorts=True/False
shorts = True
filename = 'w_score_norm_short-%s.csv' % shorts
w_score_norm = fnc.calc_weights(state_df, style='score_norm', shorts=shorts,
                                static_exposure=static_exposure, leverage=3)
w_score_norm.to_csv('Outputs/score_norm/%s' % filename)
# %%
# Static Tilts
idx_names = ['Eq Wts', 'Val Overwt', 'Mom Overwt', 'Qual Overwt',
             'Qual Underwt', 'Mom Underwt', 'Val Underwt']
static_ports = pd.DataFrame(columns=['VAL', 'MOM', 'QUAL'], index=idx_names,
                            data=[[1/3, 1/3, 1/3], [0.5, 0.25, 0.25],
                                  [0.25, 0.5, 0.25], [0.25, 0.25, 0.5],
                                  [0.4, 0.4, 0.2], [0.4, 0.2, 0.4],
                                  [0.2, 0.4, 0.4]])

# Try with shorts=True/False
shorts = True
filename = 'w_static_ports_short-%s.csv' % shorts
w_static_ports = fnc.calc_weights(state_df, style='static_tilt', shorts=shorts,
                                  static_exposure=static_exposure,
                                  rolling_window=60*12, neutral_wt=0.5,
                                  static_ports=static_ports, leverage=3)
w_static_ports.to_csv('Outputs/static_ports/%s' % filename)
# %%
# Learn score
# Try with shorts=True/False; exp_type='t'/'beta'
# leverage only used if shorts=True
shorts = True
exp_type = 't'
filename = 'w_learn_score_norm_short-%s_exp-%s.csv' % (shorts, exp_type)
exp_filename = 'exp_%s.csv' % exp_type
w_learn_score_norm, exp_df = fnc.calc_weights(
        state_df, style='learn_score_norm', shorts=shorts, rolling_window=60,
        ret_df=us_df[['MKT', 'VAL', 'MOM', 'QUAL']].copy(), exp_type=exp_type,
        leverage=3)
w_learn_score_norm.to_csv('Outputs/learn_score_norm/%s' % filename)
exp_df.to_csv('Outputs/learn_score_norm/%s' % exp_filename)
