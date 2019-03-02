"""
MFE 230K - Factor Timing Project

Authors: Ms. Allocation
"""

# Imports
import pandas as pd
import os
import matplotlib.pyplot as plt


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
    data_df = pd.concat([mkt_df.rename('MKT'), val_df.rename('VAL'),
                         mom_df.rename('MOM'), qual_df.rename('QUAL')], axis=1)
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
        data_df (pd.DataFrame)

        corrwith (str): column name wrt which correlations would be calculated

        window (int): in days

        title (str): plot title
    """
    # corrwith='MKT'; window=126; title='US_MKT'
    corrof = data_df.columns.difference([corrwith])
    plot_folder = 'Plots'
    plt.clf()
    data_df.rolling(window=window, min_periods=int(window/3)).corr(
            pairwise=True).loc[pd.IndexSlice[:, corrof], corrwith].unstack(
                    1).dropna(how='all').plot(grid=True, figsize=(12, 8),
                                              title=title)
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    plt.savefig(os.path.join(plot_folder, title + '.png'))
    plt.show()
    plt.close()
