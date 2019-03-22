"""
MFE 230K - Factor Timing Project

Authors: Ms. Allocation
"""

# Imports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from arch import arch_model
from scipy.optimize import minimize, basinhopping
from datetime import datetime


def makedir(folder, sub_folder=None):
    """
    Checks if the folder exists or not. If it doesn't, creates it.
    If sub_folder isn't None, checks folder/sub_folder

    Args:
        folder (str)

        sub_folder (str)
    """
    # folder='Plots'; sub_folder='Variance_Capture'
    if sub_folder is not None:
        folderpath = os.path.join(folder, sub_folder)
    else:
        folderpath = folder

    if not os.path.exists(folderpath):
        print('Creating folder %s' % folderpath)
        os.makedirs(folderpath)
    return None


def load_data(region):
    """
    Function to read in data according to region

    Args:
        region (str): valid values are US, JP and EU

    Returns:
        pd.DataFrame containing factors returns
    """
    # region='US'
    reg_mapper = {'US': 'USA', 'JP': 'JPN', 'EU': 'Europe'}
    if region not in reg_mapper:
        raise ValueError('region has to be one of %s'
                         % (', '.join(reg_mapper.keys())))
    data_folder = 'Data'
    filename = 'AQR_Data_Daily.xlsx'
    filepath = os.path.join(data_folder, filename)
    qual_df = pd.read_excel(filepath, sheet_name='QMJ Factors', skiprows=18,
                            parse_dates=[0], index_col=[0])[reg_mapper[region]]
    mkt_df = pd.read_excel(filepath, sheet_name='MKT', skiprows=18,
                           parse_dates=[0], index_col=[0])[reg_mapper[region]]
    mom_df = pd.read_excel(filepath, sheet_name='UMD', skiprows=18,
                           parse_dates=[0], index_col=[0])[reg_mapper[region]]
    val_df = pd.read_excel(filepath, sheet_name='HML FF', skiprows=18,
                           parse_dates=[0], index_col=[0])[reg_mapper[region]]
    rf_df = pd.read_excel(filepath, sheet_name='RF', skiprows=18,
                          parse_dates=[0], index_col=[0])['Risk Free Rate']
    data_df = pd.concat([mkt_df.rename('MKT'), val_df.rename('VAL'),
                         mom_df.rename('MOM'), qual_df.rename('QUAL'),
                         rf_df.rename('RF')], axis=1)

    # Drop dates with NaN RF
    data_df.dropna(subset=['RF'], inplace=True)
    # Drop dates with all NaNs
    data_df.dropna(how='all', inplace=True)
    # Check that returns are all valid after the first valid index
    if (data_df.apply(lambda x: x.loc[x.first_valid_index():].isnull().sum(),
                      axis=0) != 0).any():
        raise ValueError('Check the data. It has intermediate NaNs')
    # Provide basic data description
    print('Basic Description:')
    print(data_df.apply(lambda x: pd.Series(
            [x.mean(), x.std(ddof=1), x.skew(), x.kurtosis()],
            index=['Mean', 'Std Dev', 'Skew', 'Excess Kurtosis'])))
    print('\nCorrelations:')
    print(data_df.corr())
    return data_df


def rolling_corr(data_df, corrwith, window, title):
    """
    Function to plot the rolling correlations wrt the corrwith column

    Args:
        data_df (pd.DataFrame): contains returns

        corrwith (str): column name wrt which correlations would be calculated

        window (int): in days

        title (str): plot title
    """
    # corrwith='MKT'; window=126; title='US_MKT'
    corrof = data_df.columns.difference([corrwith])
    plot_folder = 'Plots'
    sub_folder = 'Factor_Corr'
    # Create Plots folder
    makedir(plot_folder, sub_folder)
    plt.clf()
    data_df.rolling(window=window, min_periods=int(window/3)).corr(
            pairwise=True).loc[pd.IndexSlice[:, corrof], corrwith].unstack(
                    1).dropna(how='all').plot(grid=True, figsize=(12, 8),
                                              title=title)
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    plt.savefig(os.path.join(plot_folder, sub_folder, title + '.png'))
    plt.ylabel('Correlations')
    plt.show()
    plt.close()


def calc_ir(data_df, bench='MKT', freq='M'):
    """
    Function to calculate rolling IRs of all factors

    Args:
        data_df (pd.DataFrame): contains returns

        bench (str): benchmark

        freq (str): provide the frequency for IR calculation

    Returns:
        ir_df (pd.DataFrame): contains IRs wrt benchmark
    """
    # data_df=us_df.copy(); bench='MKT'; freq='M'
    # bench='RF'; freq='Y'
    if freq not in ['M', 'Y']:
        raise ValueError('Only M and Y allowed as frequencies')
    irof = data_df.columns.difference([bench])
    reg_cols = [[x, bench] for x in irof]
    ann_dates_end = pd.date_range(data_df.index.min(), data_df.index.max(),
                                  freq=freq)
    ann_dates_start = pd.date_range(data_df.index.min(), data_df.index.max(),
                                    freq=freq+'S')
    date_offset = pd.offsets.YearBegin(-1) if freq == 'Y'\
        else pd.offsets.MonthBegin(-1)
    ann_dates_start = ann_dates_start.union([date_offset +
                                             data_df.index.min()])
    # Remove 2019 since it has only 1 month worth of data points for freq='Y'
    if freq == 'Y':
        ann_dates_start = ann_dates_start[:-1]
    ir_df = pd.DataFrame(columns=irof)
    for start_date, end_date in zip(ann_dates_start, ann_dates_end):
        for col in reg_cols:
            reg_df = data_df.loc[start_date:end_date, col].copy()
            # Drop NAs
            reg_df.dropna(inplace=True)
            if reg_df.empty:
                continue
            else:
                result = sm.ols('%s ~ %s' % (col[0], bench),
                                data=reg_df).fit()
                omega = np.sqrt(sum(result.resid**2))
                alpha = result.params['Intercept']
                if not np.allclose(omega, 0):
                    # Set IR = alpha/omega if sum of squared residuals > 0
                    ir_df.loc[end_date, col[0]] = float(alpha/omega)
                else:
                    ir_df.loc[end_date, col[0]] = np.nan
    # Plot IRs
    ir_df.plot(grid=True, figsize=(10, 6))
    return ir_df.astype(float), ann_dates_start, ann_dates_end


def vol_GARCH(mkt_ret, period_start):
    """
    Function to calculate the GARCH volatility for market returns

    Args:
        mkt_ret (pd.Series): market returns

        period_start (pd.DatetimeIndex): period start dates

    Returns:
        forecast_vol (pd.Series): GARCH vol prediction
    """
    # mkt_ret=us_df['MKT'].copy()
#    garch = arch_model(mkt_ret, mean='Constant', vol='GARCH', p=1, q=1,
#                       hold_back=441)
#    res = garch.fit()

    def garchFunc(coeff, y, p, q, flag=False):
        """
        Function to calculate the negative LLK for the GARCH function

        Args:
            coeff (iterable): guess values

            y (pd.Series): returns

            p, q (int)

        Returns:
            Negative LLK value
        """
        # drop the first one since theres no lagged value for it
        r = y[1:].values
        r_lagged = y.shift(1)[1:].values
        T = len(r)
        resid = r - (coeff[0] + coeff[1]*r_lagged)
        pq = max(p, q)
        # ignore t = pq since its not part of the sum
        resid_t = resid[pq:]

        # filter the h
        h_t = []
        # really important how to choose h[0]
        h_t.append(np.mean(resid ** 2))
        h_t = h_t * pq
        # calc the ht
        for i in range(pq, T):
            tmp = coeff[2]
            for j in range((3+q), (3+q+p)):
                tmp += coeff[j]*h_t[i-(j-(2+q))]
            for j in range(3, (3+q)):
                tmp += coeff[j]*(resid[i-(j-2)] ** 2)
            h_t.append(tmp)

        part_1 = -(T-pq)*np.log(2*np.pi)
        part_2 = -np.log(h_t[pq:]).sum()
        part_3 = -np.sum(np.square(resid_t)/h_t[pq:])

        condloglike = 0.5*(part_1 + part_2 + part_3)
        # because we are using a minimize function for maximizing likelihood

        if (flag):
            return pd.Series(h_t[0:1]+h_t)

        return -condloglike

    guess = np.array([.05, 0.2, .01,  0.1, 1])
    eps = np.finfo(float).eps
    bounds = [(eps, None), (None, None), (eps, None), (eps, None), (eps, None)]

    # Convert periods to monthly
    per_start = pd.date_range(period_start.min(), period_start.max(),
                              freq='MS')
    per_end = per_start + pd.offsets.MonthEnd(12)
    params = pd.DataFrame(columns=['c', 'phi', 'zeta', 'alpha', 'delta'])
    forecast_vol = pd.Series(name='GARCH_1m_FORECAST')
    # Fit GARCH and forecast vol
    for start_date, end_date in zip(per_start, per_end):
        res = minimize(fun=garchFunc, x0=guess,
                       args=(mkt_ret[start_date:end_date].copy(), 1, 1),
                       method='L-BFGS-B', bounds=bounds)
        count = 1
        while not res.success:
            min_kwargs = {'args': (mkt_ret[start_date:end_date].copy(), 1, 1),
                          'method': 'TNC', 'bounds': bounds}
            res_temp = basinhopping(func=garchFunc, x0=res.x, niter=100,
                                    minimizer_kwargs=min_kwargs)
            res = minimize(fun=garchFunc, x0=res_temp.x,
                           args=(mkt_ret[start_date:end_date].copy(), 1, 1),
                           method='L-BFGS-B', bounds=bounds)
            count += 1
            if count == 5:
                raise ValueError('Optimizer did not converge')

        params.loc[end_date] = res.x
#        next_month = [start_date+pd.offsets.MonthBegin(),
#                      start_date+pd.offsets.MonthEnd()]
        forecast_vol.loc[end_date] = np.sqrt(garchFunc(
                res.x, mkt_ret[start_date:end_date].copy(),
                1, 1, True)).iloc[-1]

    forecast_vol = forecast_vol.to_frame()
    forecast_vol.index.names = ['DATE']
    return forecast_vol


def vol_contemp(mkt_ret, period_start, period_end):
    """
    Function to calculate contemporaneous volatility

    Args:
        mkt_ret (pd.Series): market returns

        period_start (pd.DatetimeIndex): period start dates

        period_end (pd.DatetimeIndex): period end dates

    Returns:
        forecast_vol (pd.Series): GARCH vol prediction
    """
    # mkt_ret = us_df['MKT'].copy()
    vol_df = pd.DataFrame(index=period_end, columns=['Vol'])
    for start_date, end_date in zip(period_start, period_end):
        vol_df.loc[end_date] = mkt_ret[start_date:end_date].std()

    return vol_df


def load_shiller(rolling_window):
    """
    Function to read in Shiller's CAPE data


    """
    data_folder = 'Data'
    filename = 'shillerEP.csv'
    filepath = os.path.join(data_folder, filename)
    ep_df = pd.read_csv(filepath)
    # Drop rows with NaN CAPE
    ep_df.dropna(subset=['CAPE'], inplace=True)

    def dateConverter(d):
        if round(d % 1, 2) == 0.1:
            return str(d)+'0'
        else:
            return str(d)

    # Convert dates to month-end
    ep_df['Date'] = [datetime.strptime(dateConverter(d), "%Y.%m").date() +
                     pd.offsets.MonthEnd(0) for d in ep_df['Date']]
    ep_df['EP'] = 1/ep_df['CAPE'].copy()
    ep_df['EP95Pct'] = ep_df['EP'].rolling(rolling_window,
                                           min_periods=1).quantile(.95)
    ep_df['EP5Pct'] = ep_df['EP'].rolling(rolling_window,
                                          min_periods=1).quantile(.05)
    ep_df['EPmedian'] = ep_df['EP'].rolling(rolling_window,
                                            min_periods=1).quantile(.50)
    # Trim the EP values
    ep_df['EP'] = ep_df.apply(lambda x: np.clip(x['EP'], x['EP5Pct'],
                                                x['EP95Pct']), axis=1)

    ep_df.set_index('Date', inplace=True)
    ep_df.index.names = ['DATE']

    return ep_df


def cape_timing(rolling_window, neutral_wt, freq, static_wts, fac_names,
                start_date='19260731'):
    """
    Function to implement CAPE Valuation Timing

    Args:
        rolling_window (int): lookback in months used for evaluating CAPE
        levels

        neutral_wt (float): weight assigned to market assuming neutral
        valuation

        freq (str): frequency of rebalancing

        static_wts (list): provide static portfolio wts

        fac_names (list): provide list of factor names

        start_date (str): provide the start date to the weights'

    Returns:
        w (pd.DataFrame): contains weights for [MKT +  fac_names]
    """
    # rolling_window=60*12; neutral_wt=0.5; freq='M'; static_wts=[1/3]*3
    # fac_names=['qual', 'val', 'mom']; start_date='19260731'
    # Load Shiller's EP
    ep_df = load_shiller(rolling_window)
    # Calculate weights
    ep_df['w_mkt'] = neutral_wt + (ep_df['EP']-ep_df['EPmedian'])/(
            ep_df['EP95Pct']-ep_df['EP5Pct'])
    # Trim the values outside [0, 1]
    ep_df['w_mkt'] = np.clip(ep_df['w_mkt'], 0, 1)
    # Start data from start_date
    ep_df = ep_df.loc[start_date:]
    # Calculate residual weights
    ep_df['w_res'] = 1 - ep_df['w_mkt']
    # Create weights df
    w = pd.DataFrame(columns=['MKT'] + [x for x in fac_names],
                     index=ep_df.index)
    w['MKT'] = ep_df['w_mkt'].copy()
    for i, fac in enumerate(fac_names):
        col = fac
        w[col] = static_wts[i]*ep_df['w_res'].copy()

    return w


def macro_data(region, hl_smooth, us_df, garch_df):
    """
    Function to read the macro factors data

    Args:
        region (str): valid values are US, JP and EU

        hl_smooth (int): exponential halflife for smoothing growth

        us_df (pd.DataFrame): contains risk-free data

        garch_df (pd.DataFrame): 1M ahead GARCH vol forecast

    Returns:
        pd.DataFrame containing factors returns
    """
    # region='US'; hl_smooth=0
    fac_names = ['Growth', 'Inflation', 'Liquidity', 'Volatility']
    data_folder = 'Data'
    filename = 'FactSet Economic Data.xlsx'
    filepath = os.path.join(data_folder, filename)
    filename_bbg = 'BBG_Economic Data.xlsx'
    filepath_bbg = os.path.join(data_folder, filename_bbg)

    bbg_df = pd.read_excel(filepath_bbg, sheet_name='Sheet1', skiprows=6,
                           parse_dates=[0], index_col=[0]).asfreq(
                                   'M', method='ffill')
    macro_df = pd.read_excel(filepath, sheet_name='Economic Data',
                             skiprows=5, parse_dates=[0],
                             index_col=[0]).asfreq('M', method='ffill')

    if region == 'US':
        cols = macro_df.columns[1:2]
        col_names = ['Growth']
        bbg_cols = bbg_df.columns[[6, 10]]
        bbg_col_names = ['Inflation', 'LIBOR']
        # Difference inflation rate
        bbg_df[bbg_cols[0]] = bbg_df[bbg_cols[0]].diff()
    elif region == 'JP':
        cols = macro_df.columns[3:4]
        col_names = ['Growth']
        bbg_cols = bbg_df.columns[[3, 10]]
        bbg_col_names = ['LIBOR']
    elif region == 'EU':
        cols = macro_df.columns[4:5]
        col_names = ['Growth']
        bbg_cols = bbg_df.columns[10:11]
        bbg_col_names = ['LIBOR']

    macro_df = macro_df[cols].dropna(how='all')
    macro_df.columns = col_names
    macro_df.index.names = ['DATE']
    bbg_df = bbg_df[bbg_cols].dropna(how='all')/100
    bbg_df.columns = bbg_col_names
    bbg_df['LIBOR'] /= 12
    bbg_df.index.names = ['DATE']
    # Smooth values if specified
    if hl_smooth:
        macro_df = macro_df.diff().ewm(halflife=hl_smooth).mean()

    # Combine with BBG rates data
    macro_df = macro_df.merge(bbg_df, left_index=True, right_index=True,
                              how='left')
    # Combine with risk-free rate
    macro_df['RF'] = us_df['RF'].asfreq('M', method='ffill')*625/30
    macro_df['Liquidity'] = macro_df['RF'].subtract(macro_df['LIBOR'])
    # Combine with volatility
    macro_df['Volatility'] = garch_df['GARCH_1M'].asfreq('M', method='ffill')

    # Keep only the factors
    macro_df = macro_df[fac_names]
    print(macro_df.apply(lambda x: x.first_valid_index(), axis=0))

    return macro_df


def macro_states(macro_df, style, roll_window):
    """
    Function to convert macro factors into binary states

    Args:
        macro_df (pd.DataFrame): contains macro factors data

        style (str): specify method used to classify. Accepted values:
            'naive'

        roll_window (int): specify rolling window in months

    Returns:
        state_df (pd.DataFrame): macro factors classified to binary states.
        1 for up and 0 for down
    """
    # style='naive'; roll_window=60
    if style == 'naive':
        # Classify on the basis of a rolling median
        roll_median = macro_df.rolling(roll_window).median()
        state_df = macro_df >= roll_median
        state_df = state_df[pd.notnull(roll_median)].dropna(how='all')
        state_df.replace(0, -1, inplace=True)
        state_df.fillna(0, inplace=True)

    return state_df


def forecast_states(state_df, style):
    """
    Function to forecast macro factor states

    Args:
        state_df (pd.DataFrame): macro factors classified to binary states

        style (str): specify method used to classify. Accepted values:
            'constant'

    Returns:
        forecast_states_df (pd.DataFrame)
    """
    # style='constant'
    if style == 'constant':
        # Forecast next period's state equal to current period
        forecast_state_df = state_df.shift(1).dropna(how='all')

    return forecast_state_df


def calc_weights(state_df, style, shorts, **kwargs):
    """
    Function to calculate weights

    Args:
        states_df (pd.DataFrame): contains forecasts of macro factor
        weights

        style (str): specify method to calculate weights. Accepted values:
            'score_norm', 'static_tilt', 'learn_score_norm'

        shorts (bool): specify if shorting is allowed

        **kwargs (dict)

    Returns:
        w (pd.DataFrame): contains weights for [MKT +  fac_names]
    """
    # style='score_norm'; shorts=False; style='learn_score_norm'
    valid_styles = ['score_norm', 'static_tilt', 'learn_score_norm']
    if style not in valid_styles:
        raise ValueError('style has to be one of %s' % (', '.join(
                valid_styles)))
    w = pd.DataFrame(index=state_df.index,
                     columns=['MKT', 'VAL', 'MOM', 'QUAL'])

    if style == 'score_norm':
        # Calculate weights as normalized scores. Scores are provided using
        # static exposure matrix
        static_exposure = kwargs['static_exposure']
        for date in w.index:
            net_score = static_exposure @ state_df.loc[date]
            if not shorts:
                net_score = np.clip(net_score, 0, np.inf)
#            else:
#                idx = net_score.index.difference(['VAL'])
#                net_score[idx] = np.clip(net_score[idx], 0, np.inf)
            w.loc[date] = net_score/net_score.sum() if net_score.sum() else 0

    elif style == 'static_tilt':
        # Implement CAPE valuation timing and choose a static portfolio of
        # dynamic factors for tilt
        static_exposure = kwargs['static_exposure']
        rolling_window = kwargs['rolling_window']
        neutral_wt = kwargs['neutral_wt']
        static_ports = kwargs['static_ports']
        # Load Shiller's EP
        ep_df = load_shiller(rolling_window)
        # Calculate weights
        ep_df['w_mkt'] = neutral_wt + (ep_df['EP']-ep_df['EPmedian'])/(
                ep_df['EP95Pct']-ep_df['EP5Pct'])
        # Trim the values outside [0, 1]
        ep_df['w_mkt'] = np.clip(ep_df['w_mkt'], 0, 1)
        # Calculate residual weights
        ep_df['w_res'] = 1 - ep_df['w_mkt']
        # Create weights df
        w['MKT'] = ep_df['w_mkt'].copy()
        dyn_cols = w.columns.difference(['MKT'])
        for date in w.index:
            net_score = static_exposure.loc[dyn_cols] @ state_df.loc[date]
            if not shorts:
                # Allow no shorting
                net_score = np.clip(net_score, 0, np.inf)
#            else:
#                # Allow shorting of VAL
#                idx = net_score.index.difference(['VAL'])
#                net_score[idx] = np.clip(net_score[idx], 0, np.inf)

            # Choose portfolios with the highest score
            h_port = static_ports.reindex(dyn_cols, axis=1) @ net_score
            max_score = h_port.max()
            h_port = h_port.index[h_port == max_score]

            # Calc portfolio that is the mean of the highest score portfolios
            static_wt = static_ports.loc[h_port].mean(axis=0)

            w_mkt = w.loc[date, 'MKT']
            w.loc[date, dyn_cols] = (1-w_mkt)*static_wt

    elif style == 'learn_score_norm':
        # Calculate weights as normalized scores. Scores are learned in
        # a rolling window
        rolling_window = kwargs['rolling_window']
        ret_df = kwargs['ret_df']
        exp_type = kwargs['exp_type']
        # Convert to monthly returns
        valid_ret = pd.notnull(ret_df).resample('M').last()
        ret_df = ret_df.resample('M').apply(lambda x: np.product(1+x) - 1)
        ret_df = ret_df[valid_ret]
        # Create df for regression
        common_dates = state_df.index.intersection(ret_df.index)
        ret_df = ret_df.loc[common_dates]
        # Run rolling regression

        def roll_reg(ret, X, exp_type):
            exposure = pd.DataFrame(index=['MKT', 'VAL', 'MOM', 'QUAL'],
                                    columns=['Growth', 'Inflation',
                                             'Liquidity', 'Volatility'])
            for col in ret:
                reg_col = sm.OLS(ret[col], X, hasconst=True).fit()
                if exp_type == 't':
                    exposure.loc[col] = reg_col.tvalues.fillna(0)
                elif exp_type == 'beta':
                    exposure.loc[col] = reg_col.params
            return exposure

        w = pd.DataFrame(index=state_df.index[rolling_window-1:],
                         columns=['MKT', 'VAL', 'MOM', 'QUAL'])
        for date in w.index:
            dates = pd.date_range(date - pd.offsets.MonthEnd(rolling_window-1),
                                  date, freq='M')
            exposure = roll_reg(ret_df.loc[dates], state_df.loc[dates],
                                exp_type)
            net_score = exposure @ state_df.loc[date]
            if not shorts:
                net_score = np.clip(net_score, 0, np.inf)
                w.loc[date] = net_score/net_score.sum() if\
                    net_score.sum() else 0
            else:
                leverage = kwargs['leverage']
                wts = net_score/net_score.sum() if net_score.sum() else 0
                # Adjust leverage
                wts = wts/(wts.abs().sum()/(leverage-1))
                w.loc[date] = wts
#                idx = net_score.index.difference(['VAL'])
#                net_score[idx] = np.clip(net_score[idx], 0, np.inf)

    return w
