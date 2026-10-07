"""STEP 7 부가 분석 — 테스트 1회 평가(predict_test.py) 뒤에 실행한다.

1) SHAP: 트랙별 선정 모델이 트리면 그 모델을, 아니면 LightGBM(S1)을 8/31까지로 학습(seed 42, 검증에서 고른 설정)해
   테스트 기간 입력에 대한 SHAP 주효과·상호작용을 저장한다(평가 3번 영향요인용).
2) 민감도: 복사일(copied_day, 4~5월 11일)을 학습에서 뺀(가중치 0) LightGBM S1을 SEEDS 5개 × 6 fold로 다시 평가해 원래 결과와 비교
3) 재현성: 같은 seed·같은 설정으로 다시 돌려 cv_scores.csv의 원래 지표와 차이를 본다
   (나이브 기준선, LightGBM S1 F6 seed 42, NHITS S1 F6 seed 42, Chronos-2 zero-shot 트랙 B F6)
실행(프로젝트 루트에서): PYTHONHASHSEED=0 python scripts/step7_extras.py --select A=모델:시나리오 B=모델:시나리오
결과: outputs/tables/shap_*.csv, outputs/tables/sensitivity_copied_days.csv, outputs/tables/reproducibility_diff.csv
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import power_utils as pu  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings('ignore')
T = pu.OUT / 'tables'
TREES = ('lgbm', 'xgb', 'cat')
METRICS = ['mae', 'rmse', 'peak_mae', 'f1', 'pr_auc']


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def tree_params(kind, track, scenario):
    bp = pd.read_csv(T / 'tree_best_params.csv', encoding='utf-8-sig')
    return json.loads(bp[(bp['model'] == kind) & (bp['track'] == track) & (bp['scenario'] == scenario)].iloc[0]['params'])


def run_shap(df, track, selected):
    import shap
    model, scenario = selected.split(':')
    kind = model if model in TREES else 'lgbm'
    scenario = scenario if model in TREES else 'S1'
    params = tree_params(kind, track, scenario)
    X, _ = pu.make_features(df, track, scenario)
    y = df['y_obs']
    tr = df.index[(df.index < pu.TEST_START) & y.notna()]
    te = df.index[(df.index >= pu.TEST_START) & (df.index <= pu.TEST_END)]
    m = pu.make_tree_model(kind, params, pu.SEEDS[0])
    m.fit(X.loc[tr], y.loc[tr], sample_weight=pu.sample_weight(tr, pu.TEST_START, tau=params.get('tau', 'inf')))
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(X.loc[te])
    pd.DataFrame(sv, index=te, columns=X.columns).to_csv(T / f'shap_values_{track}.csv', encoding='utf-8-sig')
    inter = ex.shap_interaction_values(X.loc[te])
    mean_abs = np.abs(inter).mean(axis=0)
    rows = [{'feature_1': X.columns[i], 'feature_2': X.columns[j], 'mean_abs_interaction': mean_abs[i, j] * 2}
            for i in range(len(X.columns)) for j in range(i + 1, len(X.columns))]
    pd.DataFrame(rows).sort_values('mean_abs_interaction', ascending=False).to_csv(
        T / f'shap_interactions_{track}.csv', index=False, encoding='utf-8-sig')
    imp = pd.DataFrame({'feature': X.columns, 'mean_abs_shap': np.abs(sv).mean(axis=0),
                        'main_effect': np.abs(np.diagonal(inter, axis1=1, axis2=2)).mean(axis=0)})
    imp.assign(model=kind, scenario=scenario).sort_values('mean_abs_shap', ascending=False).to_csv(
        T / f'shap_importance_{track}.csv', index=False, encoding='utf-8-sig')
    X.loc[te].to_csv(T / f'shap_features_{track}.csv', encoding='utf-8-sig')
    base = pu.OUT / 'models' / f'final_{kind}_{track}_{scenario}_seed{pu.SEEDS[0]}'   # 최종 모델 파일(8/31까지 학습)
    if kind == 'lgbm':
        m.booster_.save_model(str(base) + '.txt')
    elif kind == 'xgb':
        m.save_model(str(base) + '.json')
    else:
        m.save_model(str(base) + '.cbm')
    log(f'SHAP 트랙 {track}: {kind} {scenario}, 테스트 {len(te)}시간 × 피처 {X.shape[1]}개')


def run_copied_days(df):
    from train_trees import fold_sets, fit_predict
    rows = []
    for track in ['A', 'B']:
        params = tree_params('lgbm', track, 'S1')
        X, _ = pu.make_features(df, track, 'S1')
        y = df['y_obs']
        sets = fold_sets(df, X, pu.make_folds('full'))
        for fs in sets:
            keep = ~df.loc[fs['tr'], 'copied_day'].to_numpy(bool)
            fs2 = {**fs, 'tr': fs['tr'][keep]}                    # 복사일 행을 학습에서 뺌(가중치 0과 같음)
            for seed in pu.SEEDS:
                _, pred, _, _ = fit_predict('lgbm', params, seed, X, y, fs2)
                sc = pu.evaluate(pd.DataFrame({'y_pred': pred}, index=fs['va']), df, fs['full_tr'])
                rows.append(sc.assign(track=track, fold=fs['fold'], seed=seed, variant='복사일 제외'))
    new = pd.concat(rows, ignore_index=True)
    cv = pd.read_csv(pu.CV_PATH, encoding='utf-8-sig')
    base = cv[(cv['model'] == 'lgbm') & (cv['scenario'] == 'S1')].assign(variant='원래(복사일 포함)')
    both = pd.concat([base, new], ignore_index=True)
    both.to_csv(T / 'sensitivity_copied_days_raw.csv', index=False, encoding='utf-8-sig')
    summ = both.groupby(['track', 'subset', 'variant'])[METRICS].mean().round(4).reset_index()
    summ.to_csv(T / 'sensitivity_copied_days.csv', index=False, encoding='utf-8-sig')
    log('민감도(복사일 제외) 완료')


def run_reproducibility(df):
    from train_trees import fold_sets, fit_predict
    import run_chronos as rc
    from train_neural import run_one
    cv = pd.read_csv(pu.CV_PATH, encoding='utf-8-sig')
    folds = pu.make_folds('full')
    f6 = folds.iloc[-1]
    rows = []

    def compare(name, track, scenario, seed, sc_new):
        old = cv[(cv['model'] == name) & (cv['track'] == track) & (cv['scenario'] == scenario) & (cv['fold'] == 'F6')
                 & (cv['seed'] == seed) & (cv['subset'] == 'all')]
        new = sc_new[sc_new['subset'] == 'all'].iloc[0]
        for m in METRICS:
            o = float(old[m].iloc[0]) if len(old) else np.nan
            rows.append({'model': name, 'track': track, 'scenario': scenario, 'fold': 'F6', 'seed': seed,
                         'metric': m, 'original': o, 'rerun': float(new[m]), 'abs_diff': abs(float(new[m]) - o)})

    for name in pu.BASELINES:                                         # 결정적 기준선
        tr, va = pu.fold_index(df.index, f6)
        compare(name, 'A', 'S0', 0, pu.evaluate(pu.baseline_predict(df, name, 'A').loc[va].to_frame(), df, tr))
    for track in ['A', 'B']:                                          # LightGBM(seed 42)
        params = tree_params('lgbm', track, 'S1')
        X, _ = pu.make_features(df, track, 'S1')
        fs = fold_sets(df, X, folds)[-1]
        _, pred, _, _ = fit_predict('lgbm', params, pu.SEEDS[0], X, df['y_obs'], fs)
        compare('lgbm', track, 'S1', pu.SEEDS[0], pu.evaluate(pd.DataFrame({'y_pred': pred}, index=fs['va']), df, fs['full_tr']))
    r = run_one(df, 'nhits', 'A', 'S1', 168, f6, pu.SEEDS[0])        # NHITS(seed 42)
    compare('nhits', 'A', 'S1', pu.SEEDS[0], r['scores'])
    rr = rc.run_unit(df, 'cov:base:B:S1:' + str(_chronos_ctx('B')) + '@F6')   # Chronos-2(결정적)
    compare('chronos2_cov', 'B', 'S1', 0, rr['tracks']['B']['scores'])
    out = pd.DataFrame(rows)
    out.to_csv(T / 'reproducibility_diff.csv', index=False, encoding='utf-8-sig')
    log(f'재현성 비교 완료: 최대 절대 차이 {out["abs_diff"].max():.2e}')


def _chronos_ctx(track):
    sc = pd.read_csv(T / 'chronos_all_scores.csv', encoding='utf-8-sig').astype({'ctx': str})
    c = sc[(sc['kind'] == 'cov') & (sc['size'] == 'base') & (sc['track'] == track)]['ctx']
    return c.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--select', nargs=2, required=True, help='A=모델:시나리오 B=모델:시나리오')
    ap.add_argument('--skip', nargs='*', default=[], help='shap copied repro 중 건너뛸 것')
    args = ap.parse_args()
    sel = dict(s.split('=') for s in args.select)
    df, _ = pu.load_data()
    if 'shap' not in args.skip:
        for track in ['A', 'B']:
            run_shap(df, track, sel[track])
    if 'copied' not in args.skip:
        run_copied_days(df)
    if 'repro' not in args.skip:
        run_reproducibility(df)


if __name__ == '__main__':
    main()
