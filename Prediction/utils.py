import pandas as pd
import os
import pickle


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
