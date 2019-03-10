import os
import pickle
from typing import List

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
sns.set(style="darkgrid")
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.facecolor'] = 'w'


def load_aqr_data(country: str = 'USA') -> pd.DataFrame:
    """
    Given the country name or `Europe`, return a table of daily factors, mkt,
    and RF for that contry as a data frame.

    Example:
    --------
                QUA      MKT         SMB          HMLFF     HMLDE  UMD  RF
    DATE
    1926-07-01   NaN  0.001003 -1.154703e-03 -3.302566e-03  ...    ...   ...
    1926-07-02   NaN  0.004409 -3.693171e-03 -4.866173e-04  ...    ...   ...
    """

    AQR_RAW_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__),
                                                os.pardir, os.pardir,
                                                'Data', 'aqr_daily.pickle'))
    with open(AQR_RAW_PATH, 'rb') as handle:
        aqr = pickle.load(handle)
        series = []
        for k, v in aqr.items():
            if k != 'RF':
                df = v[[country]]
            else:
                df = v
            df.columns = [k]
            series.append(df)

    return pd.concat(series, 1)


def build_portfolio(data: pd.DataFrame = load_aqr_data('USA').dropna(),
                    assets: List[str] = ['QUA', 'SMB', 'HMLFF', 'UMD'],
                    main_weights: List[float] = [0.4, 0.7],
                    prefix_names: List[str] = ['main', 'heavy'])\
        -> pd.DataFrame:
    """
    Build the portfolio returns based on each individual returns.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe where each column contains one factors' daily returns
    assets : List[str]
        A list of symbols that will be used to construct portfolios
    main_weights : List[float]
        A list of weights (0 ~ 1) that indicate the main weight that will be
        assigned to a factor. The rest weights will be shared evenly across
        other factors
    prefix_names : List[str]
        A list of prefix names that will be used to assign as the name of
        constructed portfolios

    Returns
    -------
    pd.DataFrame
        A data frame where each column is a reconstructed portfolio

    Example
    -------
    >>> build_portfolio(pd.DataFrame({'HML': [...], 'SMB': [...]}), ['HML', 'SMB'], [0.6], ['main']) # noqa E501
    >>>          equal_weights      main_HML     main_SMB
    DATE
    1926-07-01   NaN  0.001003 -1.154703e-03 -3.302566e-03
    1926-07-02   NaN  0.004409 -3.693171e-03 -4.866173e-04
    """

    def get_weights(w, sumup=1, items=3):
        rest_w = (sumup - w)/items

        res = []
        for i in range(items):
            weights = np.repeat(rest_w, items)
            weights[i] = w
            res.append(weights)
        return res

    def get_portfolio(w, data, prefix=None):
        name = data.columns[np.argmax(w)]
        name = prefix + '_' + name if prefix is not None else name
        df = pd.DataFrame((data * w).sum(1))
        df.columns = [name]
        return df

    df = data[assets]

    # Equal weighted case
    equal_df = pd.DataFrame(
        (df * np.repeat(1/len(assets), len(assets))).sum(1))
    equal_df.columns = ['equal_weights']

    res = []
    # Loop through each case of weights and construct portfolios accordingly
    for w, name in zip(main_weights, prefix_names):
        res.append(pd.concat([get_portfolio(w, df, name)
                              for w in get_weights(w, items=len(assets))], 1))

    return pd.concat([equal_df] + res, 1)


def plot_log_cum_returns(data: pd.DataFrame = build_portfolio())\
        -> mpl.axes._base._AxesBase:
    """
    Plot the portfolio pulled by `build_portfolio` as line_plots
    """

    returns = np.log(data + 1).cumsum()
    returns = returns.reset_index().melt(
        id_vars='DATE',
        var_name='Portfolio', value_name='Log Cum Ret')
    returns['factor'] = returns['Portfolio'].str.split('_').str[1]
    returns.loc[returns['factor'] == 'weights', 'factor'] = 'equal'
    returns['weight'] = returns['Portfolio'].str.split('_').str[0]
    returns.tail(3)

    return sns.lineplot(x="DATE", y="Log Cum Ret", hue="factor", size='weight',
                        data=returns, sizes=[1.5, 1, 1.5], alpha=0.9)
