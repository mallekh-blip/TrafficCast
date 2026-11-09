"""Safe entry point for verification, saved-model replay and results preview; training requires an explicit command."""
import argparse
from pathlib import Path
from evaluate import verify_results

def replay(project, source):
    import json
    import tempfile
    import pandas as pd
    import xgboost as xgb
    from data import file_stamp, prepare_batches, sha256
    from evaluate import compare_replay, score_batches
    from features import HORIZONS, feature_names

    inputs = project / 'outputs'
    model_dir = source / 'models'
    manifest = json.loads((source / 'run_manifest.json').read_text())
    if manifest['status'] != 'complete_baselines_passed':
        raise ValueError('Expected a completed reference run')
    if manifest['xgboost_version'] != xgb.__version__:
        raise ValueError('Use the XGBoost version recorded in the reference run')
    for year in range(2017, 2022):
        stamp = file_stamp(inputs / f'sd_{year}.pkl')
        if any(stamp[k] != manifest['input_stamps'][str(year)][k] for k in ('size', 'mtime_ns')):
            raise ValueError(f'Yearly cache changed since the reference run: {year}')
    sensors = pd.read_csv(source / 'selected_sensors.csv', dtype=str).ID.tolist()
    audited = pd.read_csv(inputs / 'audit_100_seed42' / 'selected_sensors_seed42.csv', dtype=str).ID.tolist()
    if len(set(sensors)) != 100 or sorted(sensors) != sorted(audited):
        raise ValueError('Saved and audited sensor selections differ')
    for variant in ('A', 'B'):
        for horizon in HORIZONS:
            model = model_dir / f'xgb_{variant}_h{horizon}.ubj'
            meta = json.loads(model.with_suffix('.json').read_text())
            if meta['features'] != feature_names(horizon, variant) or meta['sha256'] != sha256(model):
                raise ValueError(f'Model features or hash differ: {model}')
    lanes = pd.read_csv(inputs / 'sd_meta.csv', dtype={'ID': str}).set_index('ID').Lanes.astype(float)
    print('[PASS] 100 audited sensors and six saved XGBoost models', flush=True)
    with tempfile.TemporaryDirectory(prefix='trafficcast_replay_', dir=inputs) as temp:
        out = Path(temp)
        counts = prepare_batches(out, inputs, sensors, lanes)
        if any(counts[k] != manifest['split_counts'][k] for k in ('train', 'validation', 'test')):
            raise ValueError(f'Split counts differ: {counts}')
        result, quarters = score_batches(out, model_dir, 'test', HORIZONS)
        compare_replay(result, source / 'multiyear_results.csv', ['subset', 'method', 'horizon'])
        prepare_batches(out, inputs, sensors, lanes, stress=True)
        _, stress = score_batches(out, model_dir, 'stress', HORIZONS)
        compare_replay(pd.concat([quarters, stress]), source / 'multiyear_quarters.csv',
                       ['period', 'method', 'horizon'])
    print('[PASS] Stage 2 saved-model replay', flush=True)


def train_lgb(project):
    import json
    import shutil
    import tempfile
    from datetime import datetime, timezone
    import lightgbm
    import pandas as pd
    from data import file_stamp, prepare_batches, sha256
    from evaluate import score_batches
    from features import HORIZONS
    from models import fit_saved_lightgbm

    inputs = project / 'outputs'
    audit = inputs / 'audit_100_seed42'
    sensors = sorted(pd.read_csv(audit / 'selected_sensors_seed42.csv', dtype=str).ID.tolist())
    if len(sensors) != 100 or len(set(sensors)) != 100:
        raise ValueError('Expected exactly 100 audited sensor IDs')
    lanes = pd.read_csv(inputs / 'sd_meta.csv', dtype={'ID': str}).set_index('ID').Lanes.astype(float)
    reference = audit / 'multiyear_results_seed42.csv'
    expected = pd.read_csv(reference)
    expected = expected[expected.method.isin(
        ['Persistence', 'Same time last week', 'LightGBM A', 'LightGBM B'])]
    keys = ['subset', 'method', 'horizon']
    if len(expected) != 24 or expected.duplicated(keys).any():
        raise ValueError('Incomplete or duplicate LightGBM reference rows')
    if not expected.loc[expected.subset.eq('all days'), 'rows'].eq(10184128).all():
        raise ValueError('Reference CSV does not match the 100-sensor test row count')
    if shutil.disk_usage(inputs).free < 10 * 1024**3:
        raise RuntimeError('Stage 4 needs at least 10 GiB of free space for feature batches')
    stamps = {year: file_stamp(inputs / f'sd_{year}.pkl') for year in range(2017, 2022)}
    name = datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S%fZ')
    out = inputs / 'lgb_n100_seed42_consolidated' / name
    model_dir = out / 'models'
    model_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({'ID': sensors}).to_csv(out / 'selected_sensors.csv', index=False)
    manifest = dict(status='running', n=100, seed=42, lightgbm_version=lightgbm.__version__,
                    input_stamps=stamps, reference_sha256=sha256(reference))
    manifest_path = out / 'run_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'[output] {out}', flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix='trafficcast_lgb_', dir=inputs) as temp:
            work = Path(temp)
            counts = prepare_batches(work, inputs, sensors, lanes, include_training=True)
            if tuple(counts[k] for k in ('train', 'validation', 'test')) != (17948777, 2638611, 10184128):
                raise ValueError(f'Unexpected split counts: {counts}')
            manifest['split_counts'] = counts
            fit_saved_lightgbm(work, model_dir, counts, HORIZONS)
            results, quarters = score_batches(work, model_dir, 'test', HORIZONS, engine='lgb')
            results.to_csv(out / 'multiyear_results.csv', index=False)
            comparison = results.merge(expected, on=keys, how='outer',
                                       suffixes=('_new', '_archived'), validate='one_to_one')
            for metric in ('RMSE', 'MAE', 'NRMSE_median'):
                comparison[f'{metric}_difference'] = comparison[f'{metric}_new'] - comparison[f'{metric}_archived']
            comparison.to_csv(out / 'reference_comparison.csv', index=False)
            if not comparison.rows_new.eq(comparison.rows_archived).all():
                raise ValueError('Archived evaluation keys or counts differ')
            baselines = comparison[comparison.method.isin(['Persistence', 'Same time last week'])]
            if not baselines.RMSE_difference.abs().lt(1e-4).all():
                raise ValueError('Archived baseline RMSE differs')
            largest = comparison.RMSE_difference.abs().max()
            print(f'[comparison] Maximum absolute RMSE difference: {largest:.8f}', flush=True)
            counts['stress'] = prepare_batches(work, inputs, sensors, lanes, stress=True)['stress']
            _, stress = score_batches(work, model_dir, 'stress', HORIZONS, engine='lgb')
            pd.concat([quarters, stress]).to_csv(out / 'multiyear_quarters.csv', index=False)
        manifest.update(status='complete', max_absolute_rmse_difference=float(largest),
                        model_hashes={p.name: sha256(p) for p in model_dir.glob('*.txt')})
    except BaseException:
        manifest['status'] = 'interrupted_or_failed'
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'[done] Stage 4 LightGBM training and evaluation saved to {out}', flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['verify','preview','replay-xgb','replay-ses','train-lgb'])
    p.add_argument('--results',type=Path,default=Path(__file__).parent/'results'/'comparison.csv')
    p.add_argument('--output',type=Path,default=Path('results_preview.html'))
    p.add_argument('--project-root', type=Path)
    p.add_argument('--reference-run', type=Path)
    args=p.parse_args()
    if args.command == 'train-lgb':
        if args.project_root is None:
            p.error('train-lgb requires --project-root')
        train_lgb(args.project_root.resolve())
        return
    if args.command == 'replay-ses':
        if args.project_root is None or args.reference_run is None:
            p.error('replay-ses requires --project-root and --reference-run')
        from evaluate import replay_ses
        replay_ses(args.project_root.resolve() / 'outputs', args.reference_run.resolve())
        return
    if args.command == 'replay-xgb':
        if args.project_root is None or args.reference_run is None:
            p.error('replay-xgb requires --project-root and --reference-run')
        replay(args.project_root.resolve(), args.reference_run.resolve())
        return
    frame=verify_results(args.results)
    print(f'[PASS] {len(frame)} result rows; {frame.method.nunique()} methods; complete keys and matching counts')
    if args.command=='preview':
        from dashboard import render_results
        print('[saved]',render_results(args.results,args.output).resolve())

if __name__=='__main__':main()
