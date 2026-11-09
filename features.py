"""Feature/target construction; call per sensor batch to bound intermediates."""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import data as A
STEP=5
SPD=288
HORIZONS=[15,30,60]
FEATURES=['flow_now','flow_lag_15','flow_lag_30','flow_lag_60','flow_mean_60','hour','dow','is_weekend','is_holiday','lanes']
def longest_nan_run(df):
    out = []
    for c in df.columns:
        m = df[c].isna().to_numpy()
        if not m.any():
            out.append(0); continue
        d = np.diff(np.r_[0, m.astype(int), 0])
        out.append(int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max()))
    return pd.Series(out, index=df.columns)


def quality_flags(df):
    """Per-sensor rule table (True = rule violated)."""
    return pd.DataFrame({
        ">10% missing": df.isna().mean() > .10,
        ">10% zeros": (df == 0).mean() > .10,
        "gap >24h": longest_nan_run(df) > SPD,
        "stuck >=24h": (df.diff().abs().rolling(SPD, min_periods=SPD).max() == 0).any(),
    })


def build_table(flow, lanes_by_id):
    """One row = one sensor at one time. Features use only data at or before t;
    targets look ahead. Rows with any missing value are dropped -- gaps are
    NOT filled, so no invented data."""
    A.validate_grid(flow)
    T, S = flow.shape
    f = {"flow_now": flow,
         "flow_lag_15": flow.shift(15 // STEP),
         "flow_lag_30": flow.shift(30 // STEP),
         "flow_lag_60": flow.shift(60 // STEP),
         "flow_mean_60": flow.rolling(60 // STEP).mean()}
    for h in HORIZONS:
        k = h // STEP
        f[f"y_{h}"] = flow.shift(-k)
        f[f"seasonal_{h}"] = flow.shift(7 * SPD - k)     # same time last week
        f[f"daily_{h}"] = flow.shift(SPD - k)            # same time yesterday
    tbl = {k: v.to_numpy(dtype="float32").T.ravel() for k, v in f.items()}
    tbl = pd.DataFrame(tbl)
    t = pd.DatetimeIndex(np.tile(flow.index.to_numpy(), S))
    tbl["time"] = t
    tbl["sensor_id"] = pd.Categorical.from_codes(
        np.repeat(np.arange(S, dtype="int16"), T), categories=flow.columns.astype(str))
    tbl["hour"] = (t.hour + t.minute / 60).astype("float32")
    tbl["dow"] = np.asarray(t.dayofweek, dtype="int8")
    tbl["is_weekend"] = (np.asarray(t.dayofweek) >= 5).astype("int8")
    hol = USFederalHolidayCalendar().holidays(flow.index.min(), flow.index.max())
    tbl["is_holiday"] = t.normalize().isin(hol).astype("int8")
    lanes = lanes_by_id.reindex(flow.columns).to_numpy(dtype="float32")
    tbl["lanes"] = np.repeat(lanes, T)
    return tbl.dropna().reset_index(drop=True)


def feature_names(horizon, variant):
    if horizon not in HORIZONS or variant not in ('A','B'):
        raise ValueError('Expected horizon 15/30/60 and feature set A/B')
    return FEATURES.copy() if variant=='A' else FEATURES+[f'daily_{horizon}',f'seasonal_{horizon}']
