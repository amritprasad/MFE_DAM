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
    plt.show()
    plt.close()


def calc_ir(data_df, bench='MKT'):
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
                                  freq='Y')
    ann_dates_end = ann_dates_end.union([data_df.index.max() +
                                         pd.offsets.YearEnd()])
    ann_dates_start = pd.date_range(data_df.index.min(), data_df.index.max(),
                                    freq='YS')
    ann_dates_start = ann_dates_start.union([pd.offsets.YearBegin(-1) +
                                             data_df.index.min()])
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
    return ir_df.astype(float)
