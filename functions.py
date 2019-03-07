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
    corrs=data_df.rolling(window=window, min_periods=int(window/3)).corr(
            pairwise=True).loc[pd.IndexSlice[:, corrof], corrwith].unstack(
                    1).dropna(how='all')
    corrs.plot(grid=True, figsize=(12, 8),
                                              title=title)
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    plt.savefig(os.path.join(plot_folder, sub_folder, title + '.png'))
    plt.show()
    plt.close()
    return corrs


def calc_ir(data_df, bench='MKT',freq='Y'):
    """
    Function to calculate annual IRs of all factors

    Args:
        data_df (pd.DataFrame): contains returns

        bench (str): benchmark

    Returns:
        ir_df (pd.DataFrame): contains IRs wrt benchmark
    """
    # bench='MKT'; bench='RF'
    irof = data_df.columns.difference([bench])
    reg_cols = [[x, bench] for x in irof]
    ann_dates_end = pd.date_range(data_df.index.min(), data_df.index.max(),
                                  freq=freq)
    ann_dates_start = pd.date_range(data_df.index.min(), data_df.index.max(),
                                    freq=freq+'S')
    ann_dates_start = ann_dates_start.union([pd.offsets.YearBegin(-1) +
                                             data_df.index.min()])
    # Remove the last date since 2019 has only 1 month worth of data points
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


def vol_GARCH(mkt_ret, period_start, period_end):
    """
    Function to calculate the GARCH volatility for market returns

    Args:
        mkt_ret (pd.Series): market returns

        period_start (pd.DatetimeIndex): period start dates

        period_end (pd.DatetimeIndex): period end dates

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
    per_end = pd.date_range(period_end.min(), period_end.max(), freq='M')
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
        next_month = [end_date+pd.offsets.MonthBegin(),
                      end_date+pd.offsets.MonthEnd()]
        forecast_vol.loc[end_date] = np.sqrt(garchFunc(
                res.x, mkt_ret[next_month[0]:next_month[1]],
                1, 1, True)).iloc[-1]

    return forecast_vol
