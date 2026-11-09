"""Temporal checks and chronological partitions; preserved audited rules."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
TRAIN_START=pd.Timestamp('2017-01-01')
VAL_START=pd.Timestamp('2018-10-01')
TEST_START=pd.Timestamp('2019-01-01')
TEST_END=pd.Timestamp('2020-01-01')
GAP=pd.Timedelta(minutes=60)
DST_DAYS=['2017-03-12','2018-03-11','2019-03-10','2020-03-08','2021-03-14']
def validate_grid(flow, label="flow"):
    """Refuse ambiguous timestamps; do not guess DST or invent observations."""
    if not isinstance(flow, pd.DataFrame) or not isinstance(flow.index, pd.DatetimeIndex):
        raise ValueError(f"{label}: expected a DataFrame with a DatetimeIndex")
    idx = flow.index
    if len(idx) < 2 or idx.hasnans or not idx.is_unique or not idx.is_monotonic_increasing:
        raise ValueError(f"{label}: timestamps must be nonempty, unique and increasing (no NaT)")
    if idx.tz is not None:
        raise ValueError(f"{label}: timezone-aware data needs an explicit calendar/DST policy; do not strip timezone blindly")
    if not flow.columns.is_unique or any(not isinstance(c, str) for c in flow.columns):
        raise ValueError(f"{label}: sensor columns must be unique strings")
    delta = idx[1:] - idx[:-1]
    bad = np.flatnonzero(delta != pd.Timedelta(minutes=5))
    if len(bad):
        j = int(bad[0])
        raise ValueError(f"{label}: irregular 5-minute grid: {idx[j]} -> {idx[j+1]}. "
                         "Resolve missing/duplicate timestamps before row-based lags.")
    if ((idx.minute % 5 != 0) | (idx.second != 0) | (idx.microsecond != 0) | (idx.nanosecond != 0)).any():
        raise ValueError(f"{label}: timestamps must be aligned to 5-minute boundaries")


def mask_clock_days(flow):
    flow.loc[flow.index.normalize().isin(pd.to_datetime(DST_DAYS))] = np.nan
    return flow


def training_part(flow):
    return flow.loc[(flow.index >= TRAIN_START) & (flow.index < VAL_START - GAP)]


def assert_split(tr, va, te, val_start=VAL_START, test_start=TEST_START, test_end=TEST_END):
    for name, frame in [("training", tr), ("validation", va), ("test", te)]:
        if frame.empty:
            raise ValueError(f"{name} has no eligible rows")
    if not ((tr.time + GAP < val_start).all() and
            (va.time >= val_start).all() and (va.time + GAP < test_start).all() and
            (te.time >= test_start).all() and (te.time + GAP < test_end).all()):
        raise ValueError("Forecast targets cross a partition boundary")


def split_main(tbl):
    t = tbl.time
    tr = tbl[(t >= TRAIN_START) & (t + GAP < VAL_START)]
    va = tbl[(t >= VAL_START) & (t + GAP < TEST_START)]
    te = tbl[(t >= TEST_START) & (t + GAP < TEST_END)]
    assert_split(tr, va, te)
    return tr, va, te


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stamp(path):
    p = Path(path)
    stat = p.stat()
    return dict(path=str(p.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


import gc
import json

def load_flow(inputs, years, sensors):
    parts = [pd.read_pickle(inputs / f"sd_{year}.pkl")[sensors] for year in years]
    flow = pd.concat(parts)
    validate_grid(flow, "combined yearly cache")
    flow.loc[flow.index.normalize().isin(pd.to_datetime(DST_DAYS))] = np.nan
    return flow

def prepare_batches(out, inputs, sensors, lanes, stress=False, include_training=False):
    import numpy as np
    import pandas as pd
    from evaluate import clean_mask, holiday_affected_days, sensor_means
    import features as S
    import data as A
    if stress:
        flow = load_flow(inputs, [2019, 2020, 2021], sensors).loc['2019-12-01':]
    else:
        flow = load_flow(inputs, [2017, 2018, 2019], sensors)
    counts = {'train':0, 'validation':0, 'test':0, 'stress':0}
    means = {}
    affected = holiday_affected_days(pd.Timestamp('2019-01-01'), pd.Timestamp('2019-12-31'))
    for batch_no, start in enumerate(range(0, len(sensors), 5)):
        batch = sensors[start:start+5]
        table = S.build_table(flow[batch], lanes)
        if stress:
            frames = {'stress':table[table.time >= A.TEST_END].copy()}
        else:
            tr, va, te = A.split_main(table)
            values = sensor_means(tr, len(batch))
            means.update(dict(zip(batch, values.tolist())))
            frames = {'train':tr, 'validation':va, 'test':te}
        for split, frame in frames.items():
            counts[split] += len(frame)
            if split == 'test':
                frame = frame.copy()
                for h in S.HORIZONS:
                    frame[f'clean_{h}'] = clean_mask(frame, h, affected)
            if not include_training and split in ('train', 'validation'):
                continue
            folder = out/'batches'/split
            folder.mkdir(parents=True, exist_ok=True)
            frame.to_pickle(folder/f'batch_{batch_no:03d}.pkl')
        print(f'[prepare {"stress" if stress else "main"}] {min(start+5,len(sensors))}/{len(sensors)} sensors', flush=True)
        del table, frames, frame
        if not stress:
            del tr, va, te
        gc.collect()
    del flow
    gc.collect()
    if not stress:
        (out/'sensor_means.json').write_text(json.dumps(means))
    return counts


