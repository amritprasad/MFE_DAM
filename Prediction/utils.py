import os
import pickle
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm, tqdm_notebook

from empyrical import annual_return, max_drawdown, sharpe_ratio

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=UserWarning)
    from lightgbm import LGBMClassifier

register_matplotlib_converters()
sns.set(style="darkgrid")
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.facecolor'] = 'w'


COLOR_MAPPER = {
    'equal': '#5972A7',
    'MKT': '#AA5A58',
    'QUA': '#C78B64',
    'HMLFF': '#8C7966',
    'SMB': '#77A373',
    'UMD': '#7C75AB'
}


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
                    main_weights: List[float] = [0.7],
                    prefix_names: List[str] = ['main'],
                    include_mkt: bool = False)\
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
    include_mkt : bool, optional
        If True, the portfolio sets will also include the mkt portfolio

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

    res = pd.concat([equal_df] + res, 1)

    if include_mkt:
        res = pd.merge(res, data[['MKT']], left_index=True, right_index=True,
                       how='inner')
        res = res.rename(columns={'MKT': 'MKT_MKT'})
    return res


def _break_factor_and_weight(returns: pd.DataFrame,
                             var_name: str = 'Portfolio',
                             value_name: str = 'Log Cum Ret') -> pd.DataFrame:
    returns['factor'] = returns['Portfolio'].str.split('_').str[1]
    returns.loc[returns['factor'] == 'weights', 'factor'] = 'equal'
    returns['weight'] = returns['Portfolio'].str.split('_').str[0]
    return returns


def plot_log_cum_returns(data: pd.DataFrame = build_portfolio(),
                         start_time: Optional[str] = None)\
        -> mpl.axes._base._AxesBase:
    """
    Plot the portfolio pulled by `build_portfolio` as line_plots
    """
    if start_time is not None:
        data = data[data.index >= start_time]

    returns = np.log(data + 1).cumsum()
    returns = returns.reset_index().melt(
        id_vars='DATE', var_name='Portfolio', value_name='Log Cum Ret')
    returns = _break_factor_and_weight(returns)

    return sns.lineplot(x="DATE", y="Log Cum Ret", hue="factor", size='weight',
                        data=returns, alpha=0.9, palette=COLOR_MAPPER)


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
            left_on=['Year', 'Month'], right_on=['Year', 'Month'], how='right')

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

    def plot_log_cum_returns(self, ax: mpl.axes._base._AxesBase,
                             start_time: Optional[str] = None)\
            -> mpl.axes._base._AxesBase:
        """
        Plot the optimal log cumulative returns vs. the market returns. In
        addition, the mainly used factor will be highlighted on the graph
        """
        if ax is None:
            fig, ax = mpl.subplots(1, 1, figsize=(10, 6))

        # Get optimal cumulative returns
        df_merge = self._get_merged_returns(self.indicators)
        if start_time is not None:
            df_merge = df_merge[df_merge.index >= start_time]

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
        np.log(bench[best_cum.index] + 1).cumsum().plot(
            ax=ax, color=COLOR_MAPPER['MKT'], alpha=0.7)

        # Get the equal_weighted cumulative returns
        equal = self.portfolios['equal_weights']
        np.log(equal[best_cum.index] + 1).cumsum().plot(
            ax=ax, color=COLOR_MAPPER['equal'], alpha=1)

        sns.scatterplot(x="DATE", y="Log Cum Ret", hue='factor', s=1500,
                        marker='|', data=returns.reset_index(), ax=ax,
                        alpha=.5, palette=COLOR_MAPPER)

        return ax

    def get_labels(self) -> Tuple:
        """
        Transform the one-hot like indicator dataframe into a single column
        label dataframe

        Returns
        -------
        (pd.DataFrame, List[str])
            Single column indicator and its according names as a list
        """

        indicators = self.indicators
        labels = (indicators * list(range(indicators.shape[1]))).sum(1)\
            .astype(int)
        labels = pd.DataFrame(labels)
        labels.columns = ['labels']

        return labels, indicators.columns.tolist()

    def reverse_labels(self, labels: pd.DataFrame, label_names: List[str])\
            -> pd.DataFrame:
        """
        Reverse a label with date index into a one-hot like indicators that
        `get_best_returns` could use

        Parameters
        ----------
        labels : pd.DataFrame
            A single column indicator starting from 0 with datetime index
        label_names : List[str]
            A list of Label name where the order should match `labels`

        Returns
        -------
        pd.DataFrame:
            The one-hot like indicator dataframe
        """
        idx = labels.index
        coder = OneHotEncoder(categories='auto')
        rev_labels = coder.fit_transform(labels)\
            .toarray()

        diff = np.setdiff1d(np.arange(5.), coder.categories_[0])

        df = pd.DataFrame(rev_labels, columns=coder.categories_[0])
        df.index = idx

        if len(diff) > 0:
            for d in diff:
                df[d] = np.nan

        df = df.reindex(sorted(df.columns), axis=1)
        df.columns = label_names
        return df.replace(0, np.nan)

    def get_accuracy_performance_from_ind_probla(
            self, label: pd.DataFrame, label_probla: pd.DataFrame,
            label_names: List[str]) -> pd.DataFrame:

        threshes = np.arange(0, 1, 0.02)
        acc = []
        perc = []
        for thresh in threshes:
            ind = label_probla.dropna()[
                label_probla.dropna().max(1) >= thresh].index

            y_true = self.get_labels()[0].loc[ind].values.flatten()
            y_pred = label.loc[ind].values.flatten()

            acc.append(accuracy_score(y_true, y_pred))
            perc.append(len(y_pred) / len(label.dropna()))

        return pd.DataFrame({
            'Thresh': threshes,
            'Accuracy': acc,
            'Percentage': perc
        })

    def get_ret_from_ind_probla(self, label: pd.DataFrame,
                                label_probla: pd.DataFrame,
                                label_names: List[str],
                                thresh: float = 0.) -> pd.Series:
        ind = label_probla.dropna()[
            label_probla.dropna().max(1) >= thresh].index

        label_copy = label.dropna().copy()
        label_copy[~label_copy.dropna().index.isin(ind)] = 0

        ret = self.get_best_returns(
            self.reverse_labels(label_copy.dropna(), label_names))

        return ret


class FeatureBuilder:
    """
    A set of builders to construct the features for the machine learning layer
    classifier
    """

    macro_location = os.path.join(__file__, '..', '..', 'Data', 'macro')

    def __init__(self, country='USA'):
        self.country = country

    def _extract_macro_features(self, path=None):
        if path is None:
            path = os.path.abspath(
                os.path.join(self.macro_location,
                             self.country.lower() + '_macro.csv'))

        return pd.read_csv(path, parse_dates=True, index_col='DATE')

    def get_macro_features(self, path: Optional[str] = None) -> pd.DataFrame:
        """
        Get the macro factor features from FRED, this includes:
        - CPI
        - Production Manufacturing Index
        - Government 10Year Bond Index
        - 3 Month LIBOR rate in according currency
        - Unemployment Rate

        For each factor, will also take its difference as a feature

        Parameters
        ----------
        path : str, optional
            The file path for the macro dataset (the default is None)

        Returns
        -------
        pd.DataFrame
            The macro based features
        """

        required_features = ['cpi', 'production', 'gvnbond10y',
                             '3mlibor', 'unemployment']

        raw_macro_features = self._extract_macro_features(path=path)
        raw_macro_features = raw_macro_features[required_features]

        for feature in required_features:
            raw_macro_features[feature + '_diff'] =\
                raw_macro_features[feature].diff()

        return raw_macro_features.dropna()

    def get_aqr_features(self):
        """
        Pull AQR data directly as features, in addition, will use 30D realized
        volatility as a feature
        """

        df = load_aqr_data(self.country)

        # Get mkt vol
        vol = load_aqr_data(self.country)['MKT'].rolling(30).std()
        df['RV30D'] = vol

        df = df.dropna().resample('M').first()
        df.index = df.index - pd.offsets.MonthBegin(1)
        return df

    def get_features(self):
        """
        Get all the features and merge them together
        """

        macro_features = self.get_macro_features()
        aqr_features = self.get_aqr_features()

        return pd.merge(macro_features, aqr_features, left_index=True,
                        right_index=True, how='inner')


# ============================================================================
#                       Machine Learning Modeling Section
# ============================================================================

def in_ipynb() -> bool:
    """
    Check if the current environment is IPython Notebook

    Note, Spyder terminal is also using ZMQShell but cannot render Widget.

    Returns
    -------
    bool
        True if the current env is Jupyter notebook
    """
    try:
        zmq_status = str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>"  # noqa E501
        spyder_status = any('SPYDER' in name for name in os.environ)
        return zmq_status and not spyder_status

    except NameError:
        return False


def get_tqdm() -> Tuple:
    """
    Get proper tqdm progress bar instance based on if the current env is
    Jupyter notebook

    Note, Windows system doesn't supprot default progress bar effects

    Returns
    -------
    type, bool
        either tqdm or tqdm_notebook, the second arg will be ascii option
    """
    ascii = True if os.name == 'nt' else False

    if in_ipynb():
        # Jupyter notebook can always handle ascii correctly
        return tqdm_notebook, False
    else:
        return tqdm, ascii


class LightGBM:

    def __init__(self, params_space: Optional[Dict] = None,
                 search_args: Optional[Dict] = None) -> None:

        if params_space is None:
            self.type = 'model'
            self.model = LGBMClassifier()
        else:
            if search_args is None:
                search_args = {
                    "random_state": 123,
                    "cv": 3,
                    "n_jobs": -1,
                    "n_iter": 10
                }

            self.type = 'search'
            self.model = RandomizedSearchCV(
                estimator=LGBMClassifier(),
                param_distributions=params_space,
                **search_args
            )

    def train(self, x, y):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_probla(self, x):
        return self.model.predict_proba(x)


class RollingMethod:

    def __init__(self, rolling_bars: int = 90, predict_bars: int = 1) -> None:
        self.rolling_bars = rolling_bars
        self.predict_bars = predict_bars

    def run(self, model: LightGBM, x: pd.DataFrame, y: pd.DataFrame)\
            -> Tuple[pd.DataFrame]:

        idx = x.index

        x = x.values
        y = y.values

        if len(x) <= self.rolling_bars:
            raise Exception("RollingMethod cannot be initialzed due to size "
                            "is smaller than the rolling periods.")

        arr = np.arange(len(x) - self.rolling_bars)
        arr = arr[arr % self.predict_bars == 0] + self.rolling_bars
        arr = np.concatenate([arr, [len(x)]])

        predictions = []
        predict_problas = []

        tqdm, ascii = get_tqdm()
        for train_end, predict_end in tqdm(zip(arr[:-1], arr[1:]),
                                           total=len(arr[1:]), ascii=True):
            train_x = x[train_end - self.rolling_bars: train_end]
            train_y = y[train_end - self.rolling_bars: train_end]

            # print("Training from bar %d to bar %d..." %
            #       (train_end - self.rolling_bars, train_end - 1))
            model.train(train_x, train_y)

            predict_x = x[train_end: predict_end]

            # print("Predicting from bar %d to bar %d..." %
            #       (train_end, predict_end - 1))
            predictions.append(
                model.predict(predict_x))
            predict_problas.append(
                model.predict_probla(predict_x))

        predictions = np.concatenate(
            [np.repeat(np.nan, self.rolling_bars),
             np.concatenate(predictions)])
        predict_problas = np.concatenate(
            [np.repeat(np.nan,
                       self.rolling_bars * predict_problas[0].shape[1])
             .reshape(-1, predict_problas[0].shape[1]),
             np.concatenate(predict_problas)])

        predictions = pd.DataFrame(predictions, index=idx, columns=['labels'])
        predict_problas = pd.DataFrame(predict_problas, index=idx)

        return predictions, predict_problas
