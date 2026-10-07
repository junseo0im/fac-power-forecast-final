"""STEP 3 — LightGBM(주력)·XGBoost·CatBoost를 트랙 A·B × 시나리오 S0·S1·S2로 튜닝·평가한다.

작업 단위 = (모델, 트랙, 시나리오). 작업마다 아래를 차례로 하고 outputs/logs/trees_runs/{작업}.pkl에 저장한다.
1) 튜닝: Optuna TPESampler(seed=42), 목적 = 검증 fold 평균 MAE(seed 42 한 번, outage 제외).
   하이퍼파라미터와 최근성 가중치 τ ∈ {30, 60, ∞}를 함께 고른다. trial 30(fast 10). 테스트 구간은 쓰지 않는다.
2) 최종: 고른 설정으로 SEEDS × fold 학습·평가
   - lgbm·xgb·cat: L2(제곱오차) 점예측
   - lgbm_peakw: 피크 가중 L2(가중치 1 + α·1[y ≥ 학습 p90]). α ∈ {1, 2, 4} 중 검증 피크시간 MAE가 가장 작은 값(seed 42로 선택)
   - lgbm_q: 분위수 0.1·0.5·0.9 회귀(점예측 = q50, 교차하면 정렬)
3) 최종 후보 모델: lgbm(L2)을 테스트 이전 전체(~8/31)로 재학습해 outputs/models/에 저장(STEP 7 SHAP용, 테스트 예측은 안 함)

실행(프로젝트 루트에서). 작업을 나눠 병렬로 돌리고 마지막에 모은다:
    PYTHONHASHSEED=0 python scripts/train_trees.py --jobs lgbm:A:S1 --no-aggregate   # 작업 하나
    PYTHONHASHSEED=0 python scripts/train_trees.py --list                             # 작업 목록 출력
    PYTHONHASHSEED=0 python scripts/train_trees.py --aggregate                        # 저장된 결과 모으기
결과(모으기 후): cv_scores.csv, outputs/preds/oof_{model}_{track}_{scenario}.parquet,
    outputs/tables/tree_best_params.csv, outputs/tables/feature_importance_gain.csv, outputs/logs/optuna_trials.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import power_utils as pu  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import optuna  # noqa: E402

RUN_DIR = pu.OUT / 'logs' / 'trees_runs'
QS = [0.1, 0.5, 0.9]


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def all_jobs() -> list[str]:
    return [f'{m}:{t}:{s}' for m in pu.TREE_MODELS for t in ('A', 'B') for s in ('S0', 'S1', 'S2')]


def fold_sets(df, X, folds):
    """fold마다 (학습 인덱스, 검증 인덱스, θ·MASE용 학습 전체 인덱스, 학습 p90)을 미리 만든다."""
    y = df['y_obs']
    out = []
    for f in folds.itertuples():
        tr = df.index[(df.index < f.val_start) & y.notna()]                       # outage 타깃은 학습에서 뺌(가중치 0)
        va = df.index[(df.index >= f.val_start) & (df.index <= f.val_end)]
        assert va.max() < pu.TEST_START, '테스트 구간이 섞였습니다'
        full_tr = df.index[df.index < f.val_start]
        out.append({'fold': f.fold, 'val_start': f.val_start, 'tr': tr, 'va': va, 'full_tr': full_tr,
                    'p90': y.loc[full_tr].quantile(0.9)})
    return out


def fit_predict(kind, params, seed, X, y, fs, alpha=0.0, quantile=None):
    """한 fold를 학습해 검증 예측을 돌려준다. 가중치 = 최근성(τ) × 피크(α)."""
    w = pu.sample_weight(fs['tr'], fs['val_start'], y.loc[fs['tr']], params.get('tau', 'inf'), alpha, fs['p90'])
    model = pu.make_tree_model(kind, params, seed, quantile)
    t0 = time.time()
    model.fit(X.loc[fs['tr']], y.loc[fs['tr']], sample_weight=w)
    train_sec = time.time() - t0
    t0 = time.time()
    pred = model.predict(X.loc[fs['va']])
    infer_sec = (time.time() - t0) / (len(fs['va']) / 24)
    return model, pred, train_sec, infer_sec


def val_mae(df, fs, pred) -> float:
    yt = df.loc[fs['va'], 'y_obs'].to_numpy()
    ok = ~np.isnan(yt)
    return float(np.mean(np.abs(yt[ok] - pred[ok])))


def tune(kind, X, y, df, sets, n_trials):
    """Optuna로 하이퍼파라미터와 τ를 고른다(검증 fold 평균 MAE 최소, seed 42)."""
    def objective(trial):
        params = pu.suggest_tree_params(trial, kind)
        maes = []
        for step, fs in enumerate(sets):
            _, pred, _, _ = fit_predict(kind, params, pu.SEEDS[0], X, y, fs)
            maes.append(val_mae(df, fs, pred))
            trial.report(float(np.mean(maes)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(maes))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials)
    trials = study.trials_dataframe(attrs=('number', 'value', 'params', 'state', 'duration'))
    return study.best_params, float(study.best_value), trials


def run_variant(name, kind, params, seeds, X, y, df, sets, track, scenario, alpha=0.0, quantiles=None):
    """한 변형(lgbm, lgbm_peakw, lgbm_q 등)을 seed × fold로 학습·평가한다."""
    scores, preds, importances = [], [], []
    for fs in sets:
        for seed in seeds:
            if quantiles:
                outs = [fit_predict(kind, params, seed, X, y, fs, alpha, q) for q in quantiles]
                qp = np.sort(np.column_stack([o[1] for o in outs]), axis=1)         # 분위수 교차 방지
                pf = pd.DataFrame(qp, index=fs['va'], columns=[f'q{int(q * 100)}' for q in quantiles])
                pf['y_pred'] = pf['q50']
                train_sec, infer_sec = sum(o[2] for o in outs), sum(o[3] for o in outs)
            else:
                model, pred, train_sec, infer_sec = fit_predict(kind, params, seed, X, y, fs, alpha)
                pf = pd.DataFrame({'y_pred': pred}, index=fs['va'])
                if kind == 'lgbm':
                    gain = model.booster_.feature_importance(importance_type='gain')
                    importances.append(pd.DataFrame({'feature': X.columns, 'gain_share': gain / gain.sum(),
                                                     'fold': fs['fold'], 'seed': seed}))
            sc = pu.evaluate(pf, df, fs['full_tr'])
            scores.append(sc.assign(model=name, track=track, scenario=scenario, fold=fs['fold'], seed=seed,
                                    deterministic=False, train_sec=train_sec, infer_sec=infer_sec))
            preds.append(pf.assign(fold=fs['fold'], seed=seed))
    imp = pd.concat(importances).assign(model=name, track=track, scenario=scenario) if importances else None
    return pd.concat(scores, ignore_index=True), pd.concat(preds), imp


def run_job(job, df, mode, n_trials, n_seeds):
    kind, track, scenario = job.split(':')
    X, _ = pu.make_features(df, track, scenario)
    y = df['y_obs']
    sets = fold_sets(df, X, pu.make_folds(mode))
    seeds = pu.SEEDS[:n_seeds]
    t0 = time.time()
    best, best_mae, trials = tune(kind, X, y, df, sets, n_trials)
    log(f'{job}: 튜닝 완료 {time.time() - t0:.0f}초, 검증 MAE {best_mae:.3f}, τ={best["tau"]}')
    res = {'job': job, 'best_params': best, 'best_tune_mae': best_mae,
           'trials': trials.assign(job=job), 'variants': []}
    res['variants'].append(run_variant(kind, kind, best, seeds, X, y, df, sets, track, scenario))
    if kind == 'lgbm':
        peak_mae = {}
        for a in pu.PEAK_ALPHAS:                                                    # α 선택: 검증 피크시간 MAE
            vals = []
            for fs in sets:
                _, pred, _, _ = fit_predict(kind, best, pu.SEEDS[0], X, y, fs, alpha=a)
                yt = df.loc[fs['va'], 'y_obs'].to_numpy()
                m = ~np.isnan(yt) & (yt >= fs['p90'])
                vals.append(np.mean(np.abs(yt[m] - pred[m])))
            peak_mae[a] = float(np.mean(vals))
        alpha = min(peak_mae, key=peak_mae.get)
        res['peak_alpha'], res['peak_alpha_scores'] = alpha, peak_mae
        res['variants'].append(run_variant('lgbm_peakw', kind, best, seeds, X, y, df, sets, track, scenario, alpha=alpha))
        res['variants'].append(run_variant('lgbm_q', kind, best, seeds, X, y, df, sets, track, scenario, quantiles=QS))
        final = pu.make_tree_model(kind, best, pu.SEEDS[0])                         # 최종 후보(테스트 이전 전체)
        tr_all = df.index[(df.index < pu.TEST_START) & y.notna()]
        final.fit(X.loc[tr_all], y.loc[tr_all], sample_weight=pu.sample_weight(tr_all, pu.TEST_START, tau=best['tau']))
        (pu.OUT / 'models').mkdir(parents=True, exist_ok=True)
        final.booster_.save_model(str(pu.OUT / 'models' / f'lgbm_{track}_{scenario}.txt'))
        log(f'{job}: 피크 가중 α={alpha} (검증 피크시간 MAE {peak_mae}), 최종 후보 모델 저장')
    for sc, _, _ in res['variants']:
        m = sc.query("subset == 'all'").groupby('model')['mae'].mean()
        log(f'{job}: ' + ', '.join(f'{k} MAE {v:.3f}' for k, v in m.items()))
    res['total_sec'] = time.time() - t0
    return res


def aggregate(df):
    paths = sorted(RUN_DIR.glob('*.pkl'))
    results = [pd.read_pickle(p) for p in paths]
    missing = set(j.replace(':', '_') for j in all_jobs()) - {p.stem for p in paths}
    assert not missing, f'끝나지 않은 작업: {sorted(missing)}'
    scores, imps, params, trials = [], [], [], []
    preds = {}
    for r in results:
        kind, track, scenario = r['job'].split(':')
        for sc, pf, imp in r['variants']:
            name = sc['model'].iloc[0]
            scores.append(sc)
            preds[(name, track, scenario)] = pf
            if imp is not None:
                imps.append(imp)
        params.append({'model': kind, 'track': track, 'scenario': scenario, 'tune_mae': r['best_tune_mae'],
                       'peak_alpha': r.get('peak_alpha'), 'job_sec': round(r['total_sec']),
                       'params': json.dumps(r['best_params'], ensure_ascii=False)})
        trials.append(r['trials'])
    pu.upsert_cv_scores(pd.concat(scores, ignore_index=True))
    for (name, track, scenario), pf in preds.items():
        pu.save_oof(pf, df, name, track, scenario)
    imp = pd.concat(imps).groupby(['model', 'track', 'scenario', 'feature'])['gain_share'].agg(['mean', 'std'])
    imp.reset_index().to_csv(pu.OUT / 'tables' / 'feature_importance_gain.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(params).to_csv(pu.OUT / 'tables' / 'tree_best_params.csv', index=False, encoding='utf-8-sig')
    pd.concat(trials).to_csv(pu.OUT / 'logs' / 'optuna_trials.csv', index=False, encoding='utf-8-sig')
    log(f'모으기 완료: 작업 {len(results)}개, 모델 변형 {len(preds)}개')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fast', 'full'], default='full')
    ap.add_argument('--jobs', nargs='*', help='예: lgbm:A:S1 xgb:B:S0')
    ap.add_argument('--trials', type=int)
    ap.add_argument('--n-seeds', type=int)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--no-aggregate', action='store_true')
    args = ap.parse_args()
    if args.list:
        print('\n'.join(all_jobs()))
        return
    n_trials = args.trials or (10 if args.mode == 'fast' else 30)
    n_seeds = args.n_seeds or (3 if args.mode == 'fast' else 5)
    df, _ = pu.load_data()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for job in (args.jobs or ([] if args.aggregate else all_jobs())):
        log(f'{job}: 시작 (mode={args.mode}, trial {n_trials}, seed {n_seeds}개, 스레드 {pu.N_THREADS})')
        res = run_job(job, df, args.mode, n_trials, n_seeds)
        pd.to_pickle(res, RUN_DIR / f"{job.replace(':', '_')}.pkl")
        log(f"{job}: 완료 {res['total_sec'] / 60:.1f}분")
    if not args.no_aggregate:
        aggregate(df)


if __name__ == '__main__':
    main()
