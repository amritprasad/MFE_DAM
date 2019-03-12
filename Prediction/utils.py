import os
import pickle
from typing import List, Optional

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from empyrical import annual_return, max_drawdown, sharpe_ratio
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


def _break_factor_and_weight(returns: pd.DataFrame,
                             var_name: str = 'Portfolio',
                             value_name: str = 'Log Cum Ret') -> pd.DataFrame:
    returns['factor'] = returns['Portfolio'].str.split('_').str[1]
    returns.loc[returns['factor'] == 'weights', 'factor'] = 'equal'
    returns['weight'] = returns['Portfolio'].str.split('_').str[0]
    return returns


def plot_log_cum_returns(data: pd.DataFrame = build_portfolio())\
        -> mpl.axes._base._AxesBase:
    """
    Plot the portfolio pulled by `build_portfolio` as line_plots
    """

    returns = np.log(data + 1).cumsum()
    returns = returns.reset_index().melt(
        id_vars='DATE', var_name='Portfolio', value_name='Log Cum Ret')
    returns = _break_factor_and_weight(returns)

    return sns.lineplot(x="DATE", y="Log Cum Ret", hue="factor", size='weight',
                        data=returns, sizes=[1.5, 1, 1.5], alpha=0.9)


class PortfolioOptimizer:
    """
    The optimizer to construct the optimal portfolio based on monthly sharpe
    or other metrics
    """

    def __init__(self, portfolios: pd.DataFrame, riskfree: pd.Series,
                 market: pd.Series, country: str, freq: str = 'M') -> None:
        """
        Parameters
        ----------
        portfolios : pd.DataFrame
            Got from `build_portfolio`
        riskfree : pd.Series
            Got from `load_aqr_data`
        market : pd.Series
            Got from `load_aqr_data`
        country : str
            Will only be used for plottings etc as a meta data.
        freq : str, optional
            The `freq` used to decide the best portfolio, default to Monthly
        """
        self.portfolios = portfolios
        self.riskfree = riskfree
        self.market = market
        self.country = country
        self.freq = freq

        # assigned after `get_stats_table`
        self.stats = None
        # assigned after `get_best_indicator`
        self.indicators = None

    def get_stats_table(self, freq: Optional[str] = None,
                        metric: str = 'sharpe') -> pd.DataFrame:
        """
        Get a resampled statistic tables for each portfolio.

        Parameters
        ----------
        freq : str, optional
            Freq for resampling method (the default is 'M', monthly)
        metric : str, optional
            Metrics to be calcualted for each resample (the default is 'sharpe'
            could also be 'annual_returns' or 'max_drawdown')

        Returns
        -------
        pd.DataFrame
            The monthly (or other `freq`) sharpe ratio (or other `metric`) per
            portfolio
        """
        def sharpe(returns, rf=self.riskfree):
            """
            The default `sharpe_ratio` not come with risk-free adjustments
            """
            return sharpe_ratio((returns - rf).dropna())

        freq = self.freq if freq is None else freq

        METRICS_MAP = {
            'sharpe': sharpe,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown
        }
        metric_func = METRICS_MAP[metric]

        stats = self.portfolios.resample(freq).apply(metric_func)
        stats.index = stats.index - pd.offsets.MonthBegin()

        self.stats = stats
        return stats

    def get_best_indicators(self, stats: Optional[pd.DataFrame] = None)\
            -> pd.DataFrame:
        """
        Get a one-hot encoding indicators matrix, where only the portfolio with
        the highest stat will be marked as 1, other cells will be marked as na

        Parameters
        ----------
        stats : Optional[pd.DataFrame], optional
            Table got from `get_stats_table`.

        Returns
        -------
        pd.DataFrame
                    equal_weights  main_QUA  main_SMB  main_HMLFF  main_UMD  ..
        DATE
        1957-07-01            NaN       NaN       NaN         NaN       NaN  ..
        1957-08-01            NaN       NaN       NaN         NaN       NaN  ..
        1957-09-01            1.0       NaN       NaN         NaN       NaN  ..
        """

        stats = self.stats if stats is None else stats

        row_max = stats.max(1)
        indicators = stats.copy()
        for col in indicators.columns:
            indicators.loc[indicators[col] < row_max, col] = np.nan
            indicators.loc[indicators[col] >= row_max, col] = 1

        def validation(df):
            """
            In case there is a row that has two 1.0 indicator, if two or more
            portfolio all have the same highest stats, simply use the first one
            """
            rev = df.loc[df.sum(1) != 1, :].T
            for col in rev:
                values = rev[col].values
                values[np.where(values == 1)[0][1:]] = np.nan
                rev[col] = values
            df.loc[df.sum(1) != 1, :] = rev.T

        validation(indicators)

        self.indicators = indicators
        return indicators

    def _get_merged_returns(self, indicators) -> pd.DataFrame:

        best_strategies = indicators.reset_index()\
            .melt(id_vars='DATE', var_name='Best Portfolio')\
            .dropna().set_index('DATE').sort_index()
        best_strategies['Year'] = best_strategies.index.year
        best_strategies['Month'] = best_strategies.index.month

        portfolios = self.portfolios
        portfolios['Year'] = portfolios.index.year
        portfolios['Month'] = portfolios.index.month

        df_merge = pd.merge(
            portfolios.reset_index(), best_strategies,
            left_on=['Year', 'Month'], right_on=['Year', 'Month'], how='left')

        for p in df_merge['Best Portfolio'].unique():
            df_merge.loc[df_merge['Best Portfolio'] != p, p] = np.nan

        df_merge.set_index('DATE', inplace=True)
        return df_merge

    def get_best_returns(self, indicators: Optional[pd.DataFrame] = None)\
            -> pd.Series:
        """
        Get the optimal retunrs based on the calculated optimal indicator
        matrix.

        Parameters
        ----------
        indicators : Optional[pd.DataFrame], optional
            The opitimal one-hot like indicators calculated from
            `get_best_indicators`

        Returns
        -------
        pd.Series
            Daily returns where each day is picked as the optimal portfolio
        """

        indicators = self.indicators if indicators is None else indicators
        df_merge = self._get_merged_returns(indicators)

        return df_merge[df_merge['Best Portfolio'].unique()].sum(1)

    def plot_log_cum_returns(self, ax: mpl.axes._base._AxesBase)\
            -> mpl.axes._base._AxesBase:
        """
        Plot the optimal log cumulative returns vs. the market returns. In
        addition, the mainly used factor will be highlighted on the graph
        """
        if ax is None:
            fig, ax = mpl.subplots(1, 1, figsize=(10, 6))

        # Get optimal cumulative returns
        df_merge = self._get_merged_returns(self.indicators)
        best_ret = df_merge[df_merge['Best Portfolio'].unique()].sum(1)
        best_cum = np.log(best_ret + 1).cumsum()
        best_cum.name = 'Optimal Pick'
        best_cum.plot(ax=ax, color='#e5e5e5')

        # Capture all portfolio time-period as vertical line markers
        markers = pd.merge(
            df_merge[['Best Portfolio']], pd.DataFrame(best_cum),
            left_index=True, right_index=True, how='left')
        markers.columns = ['Portfolio', 'Log Cum Ret']
        returns = markers
        returns = _break_factor_and_weight(returns)

        # Get market cumulative returns
        bench = self.market
        bench.name = 'Mkt'
        np.log(bench[best_cum.index] + 1).cumsum().plot(ax=ax,
                                                        color='black',
                                                        alpha=0.7)

        # Get the equal_weighted cumulative returns
        equal = self.portfolios['equal_weights']
        np.log(equal[best_cum.index] + 1).cumsum().plot(ax=ax,
                                                        color='#1B4F72',
                                                        alpha=1)

        sns.scatterplot(x="DATE", y="Log Cum Ret", hue='factor', s=1500,
                        marker='|', data=returns.reset_index(), ax=ax,
                        alpha=.5)

        return ax
