"""Original evaluation metrics and holiday-unaffected filtering."""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
def holiday_affected_days(start, end):
    """Days within 3 days of a US federal holiday, plus Dec 24 - Jan 1."""
    hol = USFederalHolidayCalendar().holidays(start - pd.Timedelta(days=15),
                                              end + pd.Timedelta(days=15))
    days = set()
    for h in hol:
        days |= set(pd.date_range(h - pd.Timedelta(days=3), h + pd.Timedelta(days=3)))
    for y in range(start.year - 1, end.year + 1):
        days |= set(pd.date_range(f"{y}-12-24", f"{y + 1}-01-01"))
    return pd.DatetimeIndex(sorted(days))


def clean_mask(te, h, affected):
    """True where neither the target day nor the same day a week earlier
    (the 'same time last week' reference) is holiday-affected."""
    tgt = te["time"] + pd.Timedelta(minutes=h)
    bad = tgt.dt.normalize().isin(affected) | \
        (tgt - pd.Timedelta(days=7)).dt.normalize().isin(affected)
    return (~bad).to_numpy()

def sensor_means(df, n_sensors):
    codes = df["sensor_id"].cat.codes.to_numpy()
    s = np.bincount(codes, weights=df["flow_now"].to_numpy(), minlength=n_sensors)
    c = np.bincount(codes, minlength=n_sensors)
    return np.divide(s, c, out=np.zeros(n_sensors), where=c > 0)


def evaluate(df, preds, h, affected, train_mean, n_sensors):
    y = df[f"y_{h}"].to_numpy()
    codes = df["sensor_id"].cat.codes.to_numpy()
    masks = {"all days": np.ones(len(df), bool),
             "non-holiday": clean_mask(df, h, affected)}
    rows = []
    for subset, m in masks.items():
        for method, p in preds.items():
            e = y[m] - p[m]
            c = codes[m]
            sums = np.bincount(c, weights=e ** 2, minlength=n_sensors)
            cnt = np.bincount(c, minlength=n_sensors)
            ok = (cnt > 0) & (train_mean > 0)
            nr = np.sqrt(sums[ok] / cnt[ok]) / train_mean[ok]
            rows.append({"subset": subset, "method": method, "horizon": h,
                         "RMSE": float(np.sqrt(np.mean(e ** 2))),
                         "MAE": float(np.mean(np.abs(e))),
                         "NRMSE_median": float(np.median(nr)),
                         "rows": int(m.sum())})
    return rows


def quarter_rows(df, preds, h):
    t = df["time"] + pd.Timedelta(minutes=h)
    q = (t.dt.year * 10 + t.dt.quarter).to_numpy()
    y = df[f"y_{h}"].to_numpy()
    rows = []
    for qq in np.unique(q):
        m = q == qq
        for method, p in preds.items():
            rows.append({"period": f"{qq // 10}Q{qq % 10}", "method": method,
                         "horizon": h,
                         "RMSE": float(np.sqrt(np.mean((y[m] - p[m]) ** 2))),
                         "rows": int(m.sum())})
    return rows


def verify_results(path):
    frame=pd.read_csv(path)
    keys=['subset','method','horizon']
    needed=keys+['RMSE','MAE','NRMSE_median','rows']
    if not set(needed)<=set(frame): raise ValueError('Missing result columns')
    if frame.duplicated(keys).any(): raise ValueError('Duplicate result keys')
    values=frame[['RMSE','MAE','NRMSE_median','rows']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<0).any(): raise ValueError('Invalid metrics/counts')
    if (frame.rows<=0).any() or (frame.rows%1!=0).any(): raise ValueError('Invalid row counts')
    expected={(s,m,h) for s in ['all days','non-holiday'] for m in frame.method.unique() for h in [15,30,60]}
    if set(map(tuple,frame[keys].to_numpy()))!=expected: raise ValueError('Incomplete result combinations')
    if (frame.groupby(['subset','horizon']).rows.nunique()!=1).any(): raise ValueError('Unequal evaluation counts')
    return frame


import json

def score_batches(out, model_dir, split, horizons, models=True, engine="xgb"):
    import numpy as np
    import pandas as pd
    means = json.loads((out/'sensor_means.json').read_text())
    totals, quarterly = {}, {}
    loaded = {}
    if models:
        for h in horizons:
            for variant in ['A','B']:
                if engine == 'lgb':
                    from lightgbm import Booster
                    loaded[h, variant] = Booster(
                        model_file=str(model_dir / f'lgb_{variant}_h{h}.txt'))
                    continue
                import xgboost as xgb
                path = model_dir/f'xgb_{variant}_h{h}.ubj'
                meta = json.loads(path.with_suffix('.json').read_text())
                model = xgb.Booster()
                model.load_model(path)
                model.set_param({'nthread':4})
                loaded[h, variant] = model, meta
    for path in sorted((out/'batches'/split).glob('*.pkl')):
        df = pd.read_pickle(path)
        codes = df.sensor_id.cat.codes.to_numpy()
        ids = df.sensor_id.cat.categories
        ns = len(ids)
        train_means = np.array([means[str(s)] for s in ids])
        for h in horizons:
            y = df[f'y_{h}'].to_numpy(dtype=float)
            preds = {'Persistence':df.flow_now.to_numpy(),
                     'Same time last week':df[f'seasonal_{h}'].to_numpy()}
            if models:
                for variant in ['A','B']:
                    if engine == 'lgb':
                        from features import feature_names
                        preds[f'LightGBM {variant}'] = loaded[h, variant].predict(
                            df[feature_names(h, variant)], num_threads=4)
                        continue
                    model, meta = loaded[h, variant]
                    preds[f'XGBoost {variant}'] = model.inplace_predict(
                        df[meta['features']].to_numpy(dtype=np.float32),
                        iteration_range=(0, int(meta['best_iteration'])+1))
            masks = {'all days':np.ones(len(df),dtype=bool)}
            if split == 'test':
                masks['non-holiday'] = df[f'clean_{h}'].to_numpy(dtype=bool)
            for subset, mask in masks.items():
                cnt = np.bincount(codes[mask], minlength=ns)
                ok = (cnt>0)&(train_means>0)
                for name, p in preds.items():
                    e = y[mask]-p[mask]
                    sse = np.bincount(codes[mask],weights=e**2,minlength=ns)
                    rec = totals.setdefault((subset,name,h),[0,0.,0.,[]])
                    rec[0] += len(e); rec[1] += float(np.sum(e**2)); rec[2] += float(np.sum(abs(e)))
                    rec[3].extend((np.sqrt(sse[ok]/cnt[ok])/train_means[ok]).tolist())
            target = df.time+pd.Timedelta(minutes=h)
            quarters = (target.dt.year*10+target.dt.quarter).to_numpy()
            for q in np.unique(quarters):
                mask = quarters == q
                for name, p in preds.items():
                    e = y[mask]-p[mask]
                    rec = quarterly.setdefault((f'{q//10}Q{q%10}',name,h),[0,0.])
                    rec[0] += len(e); rec[1] += float(np.sum(e**2))
        print(f'[evaluate {split}] {path.name}',flush=True)
    results = pd.DataFrame([dict(subset=k[0],method=k[1],horizon=k[2],rows=v[0],
        RMSE=np.sqrt(v[1]/v[0]), MAE=v[2]/v[0],NRMSE_median=np.median(v[3])) for k,v in totals.items()])
    quarters = pd.DataFrame([dict(period=k[0],method=k[1],horizon=k[2],rows=v[0],
        RMSE=np.sqrt(v[1]/v[0])) for k,v in quarterly.items()])
    return results, quarters


def compare_replay(actual, reference, keys):
    expected = pd.read_csv(reference)
    merged = actual.merge(expected, on=keys, how='outer', indicator=True,
                          suffixes=('_new', '_ref'), validate='one_to_one')
    if not merged['_merge'].eq('both').all() or not merged.rows_new.eq(merged.rows_ref).all():
        raise ValueError(f'Result keys or counts differ from {reference}')
    for metric in ('RMSE', 'MAE', 'NRMSE_median'):
        if metric in expected and not np.allclose(
            merged[f'{metric}_new'], merged[f'{metric}_ref'], rtol=0, atol=1e-6
        ):
            raise ValueError(f'{metric} differs from {reference}')
    print(f'[PASS] {len(merged)} rows match {reference.name}', flush=True)


def replay_ses(inputs, reference):
    from data import load_flow, split_main
    from features import build_table, HORIZONS
    from models import at_rows, levels

    manifest = json.loads((reference / 'ses_manifest.json').read_text())
    if (manifest['n'], manifest['seed']) != (100, 42):
        raise ValueError('Expected the 100-sensor seed-42 SES run')
    sensors = pd.read_csv(reference / 'ses_selected_sensors.csv', dtype=str).ID.tolist()
    audited = pd.read_csv(inputs / 'audit_100_seed42' / 'selected_sensors_seed42.csv', dtype=str).ID.tolist()
    if len(sensors) != 100 or len(set(sensors)) != 100 or sorted(sensors) != sorted(audited):
        raise ValueError('Saved and audited sensor selections differ')
    chosen = {h: float(manifest['alpha_by_horizon'][str(h)]) for h in HORIZONS}
    if any(not 0 < alpha <= 1 for alpha in chosen.values()):
        raise ValueError('Invalid SES alpha')
    batch_size = int(manifest['sensor_batch_size'])
    if batch_size < 1:
        raise ValueError('Invalid sensor batch size')
    lanes = pd.read_csv(inputs / 'sd_meta.csv', dtype={'ID': str}).set_index('ID').Lanes.astype(float)
    batches = [sensors[i:i + batch_size] for i in range(0, len(sensors), batch_size)]
    flow = load_flow(inputs, [2017, 2018, 2019], sensors)
    affected = holiday_affected_days(pd.Timestamp('2019-01-01'), pd.Timestamp('2019-12-31'))
    totals = {}
    split_counts = np.zeros(3, dtype='int64')
    for i, batch in enumerate(batches, 1):
        part = flow[batch]
        tr, va, te = split_main(build_table(part, lanes))
        split_counts += [len(tr), len(va), len(te)]
        means = sensor_means(tr, len(batch))
        codes = te['sensor_id'].cat.codes.to_numpy()
        states = {a: at_rows(levels(part, a), te) for a in set(chosen.values())}
        for h in HORIZONS:
            y = te[f'y_{h}'].to_numpy(dtype=float)
            preds = {'Persistence': te['flow_now'].to_numpy(),
                     'Same time last week': te[f'seasonal_{h}'].to_numpy(),
                     'SES': states[chosen[h]]}
            masks = {'all days': np.ones(len(te), dtype=bool),
                     'non-holiday': clean_mask(te, h, affected)}
            for subset, mask in masks.items():
                c = codes[mask]
                counts = np.bincount(c, minlength=len(batch))
                ok = (counts > 0) & (means > 0)
                for method, pred in preds.items():
                    error = y[mask] - pred[mask]
                    squared = error ** 2
                    sse = np.bincount(c, weights=squared, minlength=len(batch))
                    entry = totals.setdefault((subset, method, h), [0, 0., 0., []])
                    entry[0] += len(error)
                    entry[1] += np.sum(squared)
                    entry[2] += np.sum(np.abs(error))
                    entry[3].extend(np.sqrt(sse[ok] / counts[ok]) / means[ok])
        print(f'[test] sensor batch {i}/{len(batches)}', flush=True)
        del part, tr, va, te, codes, states, preds, masks, y, error, squared, c, pred
    print(f'[split] train {split_counts[0]:,}; validation {split_counts[1]:,}; test {split_counts[2]:,}')
    rows = [
        {'subset': subset, 'method': method, 'horizon': h, 'RMSE': np.sqrt(sse / count),
         'MAE': sae / count, 'NRMSE_median': np.median(normalized), 'rows': count}
        for (subset, method, h), (count, sse, sae, normalized) in totals.items()
    ]
    compare_replay(pd.DataFrame(rows), reference / 'ses_results.csv',
                   ['subset', 'method', 'horizon'])
    print('[PASS] Stage 3 SES replay', flush=True)
