"""Existing model factories, fitting rules and SES primitives. No LSTM implementation."""
import numpy as np
import pandas as pd
SEED=42
def make_xgb():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                            tree_method="hist", early_stopping_rounds=30,
                            n_jobs=-1, random_state=SEED), "XGBoost", "xgb"
    except ImportError as exc:
        raise ImportError("Install xgboost in the active project environment; no stand-in model is permitted") from exc


def make_lgbm():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=600, learning_rate=0.05,
                             num_leaves=31, n_jobs=-1, random_state=SEED,
                             verbose=-1), "LightGBM", "lgb"
    except ImportError as exc:
        raise ImportError("Install lightgbm in the active project environment; no stand-in model is permitted") from exc


def fit(model, kind, Xtr, ytr, Xva, yva):
    if kind == "xgb":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif kind == "lgb":
        import lightgbm as lgb
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    else:
        model.fit(Xtr, ytr)


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p): return float(np.mean(np.abs(y - p)))

def levels(flow, alpha):
    # Initial state = first finite reading. Missing readings do not update state.
    # State is carried over outages; forecasts at missing issue readings are masked.
    smoothed = flow.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
    return smoothed.where(flow.notna())


def at_rows(state, df):
    ri = state.index.get_indexer(pd.DatetimeIndex(df['time']))
    ci = state.columns.astype(str).get_indexer(df['sensor_id'].astype(str))
    if (ri < 0).any() or (ci < 0).any():
        raise ValueError('Sensor/time lookup failed')
    p = state.to_numpy()[ri, ci]
    if not np.isfinite(p).all():
        raise ValueError('SES predictions unavailable on original evaluation rows')
    return p


def fit_saved_lightgbm(work, destination, counts, horizons):
    from features import feature_names

    def matrix(split, columns, horizon):
        x = np.empty((counts[split], len(columns)), dtype=np.float32)
        y = np.empty(counts[split], dtype=np.float32)
        offset = 0
        for path in sorted((work / 'batches' / split).glob('*.pkl')):
            batch = pd.read_pickle(path)
            end = offset + len(batch)
            x[offset:end] = batch[columns].to_numpy(dtype=np.float32)
            y[offset:end] = batch[f'y_{horizon}']
            offset = end
        if offset != counts[split]:
            raise ValueError(f'Unexpected {split} row count')
        return pd.DataFrame(x, columns=columns, copy=False), y

    for horizon in horizons:
        for variant in ('A', 'B'):
            columns = feature_names(horizon, variant)
            train, y_train = matrix('train', columns, horizon)
            valid, y_valid = matrix('validation', columns, horizon)
            model, _, kind = make_lgbm()
            print(f'[fit] LightGBM {variant} | {horizon} min', flush=True)
            fit(model, kind, train, y_train, valid, y_valid)
            path = destination / f'lgb_{variant}_h{horizon}.txt'
            model.booster_.save_model(str(path), num_iteration=model.best_iteration_)
            print(f'[saved model] {path.name} | best iteration {model.best_iteration_}', flush=True)
            del model, train, valid, y_train, y_valid
