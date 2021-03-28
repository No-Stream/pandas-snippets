import pandas as pd


def py_dt_to_pd_dt(ser: pd.Series, tz: str = "UTC") -> pd.Series:
    """
    Weirdly, we lack some tz ops on Series, so we have to coerce to index and then back to Series.
    """
    orig_index = ser.index.copy()
    ts = (
        pd.DatetimeIndex(pd.to_datetime(ser.apply(lambda x: pd.Timestamp(x)), utc=True))
        .tz_convert(tz)
        .to_series()
    )
    ts.index = orig_index  # reindex throwing errs
    return ts
