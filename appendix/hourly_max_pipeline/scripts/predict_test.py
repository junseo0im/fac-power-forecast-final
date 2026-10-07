"""STEP 7 — 테스트 구간(2021-09-01~09-14) 1회 평가용 예측. 이 프로젝트에서 테스트 구간을 쓰는 유일한 스크립트다.

반드시 최종 선정 규칙을 기록한 뒤 한 번만 실행한다. 여기서는 아무것도 고르거나 맞추지 않는다:
- 모든 모델은 8/31까지의 데이터로 다시 학습하되, 하이퍼파라미터·step·epoch는 검증에서 정한 값을 그대로 쓴다.
  트리 = Optuna 최적 설정(seed 5개 평균), 딥러닝 = fold별 best step 중앙값(early stopping 없음, seed 3개 평균),
  가이드북 RNN = best epoch 중앙값(164, seed 3개 평균), 통계 모델·Chronos-2 zero-shot = 같은 방식으로 예측.
- 앙상블·컨포멀·확률보정·임계값은 STEP 6에서 6 fold 전체로 맞춘 설정(outputs/models/step6_final_config.pkl)을 적용한다.
실행(프로젝트 루트에서): PYTHONHASHSEED=0 python scripts/predict_test.py --models all > outputs/logs/predict_test.log 2>&1
결과: outputs/preds/oof_test_{model}_{track}_{scenario}.parquet, outputs/tables/test_scores.csv
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('KERAS_BACKEND', 'torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import power_utils as pu  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings('ignore')
TEST = SimpleNamespace(fold='TEST', val_start=pu.TEST_START, val_end=pu.TEST_END)
RUN_DIR = pu.OUT / 'logs' / 'test_runs'
S1_MODELS = ['lgbm', 'lgbm_q', 'lgbm_peakw', 'xgb', 'cat', 'nhits', 'lstm', 'chronos2_cov', 'arima']
S0_MODELS = ['naive_last', 'naive_week', 'naive_4wk', 'guidebook_rnn', 'guidebook_rf', 'mstl', 'chronos2_zs']


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def test_index(df):
    return df.index[(df.index >= pu.TEST_START) & (df.index <= pu.TEST_END)]


def train_index(df):
    return df.index[df.index < pu.TEST_START]


# --- 모델별 테스트 예측 ---------------------------------------------------------------------
def pred_naive(df, name, track):
    return pu.baseline_predict(df, name, track).loc[test_index(df)].to_frame()


def pred_trees(df, model, track, scenario):
    bp = pd.read_csv(pu.OUT / 'tables' / 'tree_best_params.csv', encoding='utf-8-sig')
    kind = model.split('_')[0]
    row = bp[(bp['model'] == kind) & (bp['track'] == track) & (bp['scenario'] == scenario)].iloc[0]
    params = json.loads(row['params'])
    X, _ = pu.make_features(df, track, scenario)
    y = df['y_obs']
    tr = df.index[(df.index < pu.TEST_START) & y.notna()]
    te = test_index(df)
    p90 = y.loc[train_index(df)].quantile(0.9)
    alpha = float(row['peak_alpha']) if model == 'lgbm_peakw' else 0.0
    qs = [0.1, 0.5, 0.9] if model == 'lgbm_q' else [None]
    outs = []
    for seed in pu.SEEDS:
        w = pu.sample_weight(tr, pu.TEST_START, y.loc[tr], params.get('tau', 'inf'), alpha, p90)
        cols = []
        for q in qs:
            m = pu.make_tree_model(kind, params, seed, q)
            m.fit(X.loc[tr], y.loc[tr], sample_weight=w)
            cols.append(m.predict(X.loc[te]))
        outs.append(np.sort(np.column_stack(cols), axis=1))
    P = np.mean(outs, axis=0)
    if model == 'lgbm_q':
        pf = pd.DataFrame(P, index=te, columns=['q10', 'q50', 'q90'])
        pf['y_pred'] = pf['q50']
        return pf
    return pd.DataFrame({'y_pred': P[:, 0]}, index=te)


def pred_guidebook_rf(df):
    from sklearn.ensemble import RandomForestRegressor
    X, _ = pu.guidebook_rf_features(df)
    y = df['y_obs']
    tr = df.index[X.notna().all(axis=1) & y.notna() & (df.index < pu.TEST_START)]
    te = test_index(df)
    ps = [RandomForestRegressor(max_depth=20, random_state=s, n_jobs=pu.N_THREADS).fit(X.loc[tr], y.loc[tr]).predict(X.loc[te])
          for s in pu.SEEDS]
    return pd.DataFrame({'y_pred': np.mean(ps, axis=0)}, index=te)


def pred_guidebook_rnn(df, track):
    import keras
    epochs = json.loads((pu.OUT / 'logs' / 'guidebook_rnn_summary.json').read_text(encoding='utf-8'))['test_refit_epochs']
    y_in = pu.input_series(df)
    lags, yh = pu.lag_matrix(y_in), y_in.to_numpy()
    target = df['y_obs'].to_numpy()
    idx = df.index
    tr = (np.arange(len(idx)) >= pu.RNN_LAGS) & ~np.isnan(target) & (idx < pu.TEST_START)
    te = (idx >= pu.TEST_START) & (idx <= pu.TEST_END)
    preds = []
    for seed in pu.SEEDS[:3]:
        pu.set_seed(seed)
        keras.utils.set_random_seed(seed)
        sc = pu.GuidebookScaler().fit(target[tr], lags[tr])
        model = pu.build_guidebook_rnn()
        model.fit(sc.transform_lags(lags[tr]).reshape(-1, 24, 7), sc.transform_y(target[tr]), epochs=epochs,
                  batch_size=32, shuffle=True, verbose=0)
        f = lambda L: sc.inverse_y(model.predict(sc.transform_lags(L).reshape(-1, 24, 7), batch_size=1024, verbose=0).ravel())
        if track == 'A':
            preds.append(f(lags[te]))
        else:
            o = np.where(te & (df['hour'].to_numpy() == 0))[0]
            preds.append(pu.recursive_forecast(f, np.stack([yh[i - pu.RNN_LAGS:i] for i in o]), 24).ravel())
    return pd.DataFrame({'y_pred': np.mean(preds, axis=0)}, index=test_index(df))


def pred_stats(df, name, track, scenario):
    """MSTL·AutoARIMA: 8/31까지로 한 번 학습하고 테스트 구간은 실제값으로 상태만 갱신하며 예측(검증과 같은 방식)."""
    from statsforecast import StatsForecast
    from train_stats import make_model, TRACK_CV
    upto = df.index <= pu.TEST_END
    sdf = pd.DataFrame({'unique_id': 'y', 'ds': df.index[upto], 'y': pu.input_series(df)[upto].to_numpy()})
    if name == 'arima':
        sdf = pd.concat([sdf, pu.arima_exog(df, scenario, train_index(df)).loc[upto].reset_index(drop=True)], axis=1)
    h = TRACK_CV[track]['h']
    cv = StatsForecast(models=[make_model(name)], freq='h', n_jobs=1).cross_validation(
        df=sdf, n_windows=len(test_index(df)) // h, refit=False, level=[80], **TRACK_CV[track]).set_index('ds')
    return pd.DataFrame({'y_pred': cv[name], 'q10': cv[f'{name}-lo-80'], 'q90': cv[f'{name}-hi-80']}).loc[test_index(df)]


def pred_neural(df, kind, track, scenario):
    import torch
    from neuralforecast import NeuralForecast
    from train_neural import build_frame, MODELS, COMMON, HIST, TRACK_CV as NCV
    runs = pd.read_csv(pu.OUT / 'tables' / 'neural_runs.csv', encoding='utf-8-sig')
    steps = int(runs[(runs['model'] == kind) & (runs['track'] == track) & (runs['scenario'] == scenario)]['best_step'].median())
    frame, futr, (mu, sd) = build_frame(df, scenario, train_index(df))
    frame = frame[df.index <= pu.TEST_END].reset_index(drop=True)
    cls, cfg = MODELS[kind]
    h, step = NCV[track]['h'], NCV[track]['step_size']
    common = {**COMMON, 'early_stop_patience_steps': -1, 'val_check_steps': 10 ** 6}
    preds = []
    for seed in pu.SEEDS[:3]:
        pu.set_seed(seed)
        torch.set_num_threads(pu.N_THREADS)
        m = cls(h=h, input_size=168, futr_exog_list=futr, hist_exog_list=HIST, random_seed=seed, alias=kind,
                **common, **{**cfg, 'max_steps': max(steps, 50)})
        cv = NeuralForecast(models=[m], freq='h').cross_validation(df=frame, n_windows=len(test_index(df)) // h,
                                                                    step_size=step, val_size=0, refit=False)
        preds.append(cv.set_index('ds')[kind].to_numpy() * sd + mu)
    return pd.DataFrame({'y_pred': np.mean(preds, axis=0)}, index=test_index(df))


def pred_chronos(df, model, track, scenario):
    import run_chronos as rc
    sc = pd.read_csv(pu.OUT / 'tables' / 'chronos_all_scores.csv', encoding='utf-8-sig').astype({'ctx': str})
    zs = sc[(sc['kind'] == 'zs') & (sc['subset'] == 'all') & (sc['track'] == track)]
    ctx = zs.groupby('ctx')['mae'].mean().idxmin()                     # 검증에서 고른 문맥 길이 그대로
    if model == 'chronos2_cov':
        cc = sc[(sc['kind'] == 'cov') & (sc['size'] == 'base') & (sc['track'] == track)]['ctx']
        ctx = str(cc.iloc[0]) if len(cc) else ctx
    h = 1 if track == 'A' else 24
    te = np.where((df.index >= pu.TEST_START) & (df.index <= pu.TEST_END))[0]
    starts = te if track == 'A' else te[df['hour'].to_numpy()[te] == 0]
    cov, known = (None, None) if model == 'chronos2_zs' else rc.covariates(df, scenario, train_index(df))
    items = rc.make_items(df['y_obs'].to_numpy(float), cov, known, starts, ctx, h)
    pf, _ = rc.predict_frame(rc.pipeline('base'), items, starts, h, df)
    return pf


def predict_model(df, model, track, scenario):
    if model.startswith('naive'):
        return pred_naive(df, model, track)
    if model in ('lgbm', 'lgbm_q', 'lgbm_peakw', 'xgb', 'cat'):
        return pred_trees(df, model, track, scenario)
    if model == 'guidebook_rf':
        return pred_guidebook_rf(df)
    if model == 'guidebook_rnn':
        return pred_guidebook_rnn(df, track)
    if model in ('mstl', 'arima'):
        return pred_stats(df, model, track, scenario)
    if model in ('nhits', 'lstm'):
        return pred_neural(df, model, track, scenario)
    if model.startswith('chronos2'):
        return pred_chronos(df, model, track, scenario)
    raise ValueError(model)


# --- STEP 6 설정을 그대로 적용한 앙상블·확률 ----------------------------------------------------
def pred_ensembles(df, track, comp):
    """comp: {모델: 테스트 예측 표}. STEP 6에서 6 fold 전체로 맞춘 가중치·CQR·확률 층을 그대로 적용한다."""
    from step6_ensemble import train_classifier, stack_feats
    cfg = pd.read_pickle(pu.OUT / 'models' / 'step6_final_config.pkl')[track]
    te = test_index(df)
    P = pd.DataFrame({m: comp[m]['y_pred'] for m in cfg['candidates']}, index=te)
    out = {'ens_nnls': pd.DataFrame({'y_pred': P.to_numpy() @ np.array([cfg['nnls_weights'][m] for m in P])}, index=te),
           'ens_top3': P[cfg['top3']].mean(axis=1).to_frame('y_pred')}
    if 'nnls_hour_weights' in cfg:
        e = np.zeros(len(te))
        blk = df.loc[te, 'hour'].to_numpy() // 4
        for b, w in cfg['nnls_hour_weights'].items():
            e[blk == int(b)] = P[blk == int(b)].to_numpy() @ np.array([w[m] for m in P])
        out['ens_nnls_hour'] = pd.DataFrame({'y_pred': e}, index=te)
    qsrc = [m for m in ['lgbm_q', 'chronos2_cov'] if m in comp]
    ens_q = sum(comp[m][['q10', 'q50', 'q90']] for m in qsrc) / len(qsrc)
    ens_q = pd.DataFrame(np.sort(ens_q.to_numpy(), axis=1), index=te, columns=['q10', 'q50', 'q90'])
    cqr = pu.conformal_adjust(cfg['cqr_cal'], ens_q, np.zeros(len(cfg['cqr_cal']), int), np.zeros(len(te), int))
    theta = df.loc[train_index(df), 'y_obs'].quantile(0.95)
    X, _ = pu.make_features(df, track, 'S1')
    s = {'idx': te, 'theta': theta, 'ens_q': ens_q,
         'p1q': pd.Series(pu.peak_prob_from_quantiles(ens_q, theta), index=te),
         'p2': train_classifier(df, X, TEST, theta, True, pu.SEEDS), 'e_ref': P.mean(axis=1)}
    layer = cfg['prob_layer']
    raw = {'p1q': s['p1q'].to_numpy(), 'p2': s['p2'].to_numpy(),
           'stack': layer['stack'].predict_proba(stack_feats(s))[:, 1]}
    if 'chronos2_cov' in comp and all(c in comp['chronos2_cov'] for c in pu.Q_COLS):
        raw['p1'] = pu.peak_prob_from_quantiles(comp['chronos2_cov'], theta)
    # 모든 확률 경로를 STEP 6 보정기로 보정하고, 각 경로의 F1 최대 임계값(6 fold 전체로 고른 값)을 함께 저장
    alarm = pd.DataFrame({f'p_{k}': layer['iso'][k].predict(v) for k, v in raw.items() if k in layer['iso']}, index=te)
    alarm['theta'] = theta
    thr_all = {k: layer['thr'][(k, '단순 피크', 'F1 최대')] for k in raw if (k, '단순 피크', 'F1 최대') in layer['thr']}
    alarm.to_parquet(pu.OUT / 'preds' / f'test_alarm_{track}.parquet')
    (pu.OUT / 'tables' / f'test_alarm_thresholds_{track}.json').write_text(json.dumps(thr_all, indent=2), encoding='utf-8')
    out['ens_final'] = pd.DataFrame({'y_pred': out['ens_nnls']['y_pred'], 'q10': cqr['q10'], 'q50': ens_q['q50'],
                                     'q90': cqr['q90'], 'p_peak': alarm['p_stack']}, index=te)
    return out, thr_all['stack']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=['all'])
    ap.add_argument('--tracks', nargs='*', default=['A', 'B'])
    ap.add_argument('--no-ensemble', action='store_true', help='일부 모델만 예측(병렬용). 앙상블·채점 파일은 마지막 전체 실행에서 만든다')
    args = ap.parse_args()
    df, _ = pu.load_data()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(m, 'S1') for m in S1_MODELS] + [(m, 'S0') for m in S0_MODELS]
    if args.models != ['all']:
        jobs = [j for j in jobs if j[0] in args.models]
    rows = []
    for track in args.tracks:
        comp = {}
        for model, scenario in jobs:
            if model == 'guidebook_rf' and track == 'B':
                continue
            path = RUN_DIR / f'{model}_{track}_{scenario}.pkl'
            if path.exists():
                pf = pd.read_pickle(path)
            else:
                t0 = time.time()
                pf = predict_model(df, model, track, scenario)
                pd.to_pickle(pf, path)
                log(f'{model} {track} {scenario}: {time.time() - t0:.0f}초')
            comp[model] = pf
            rows.append(pu.evaluate(pf, df, train_index(df)).assign(model=model, track=track, scenario=scenario))
        if args.no_ensemble:
            continue
        ens, thr = pred_ensembles(df, track, comp)
        for name, pf in ens.items():
            rows.append(pu.evaluate(pf, df, train_index(df), threshold=thr if name == 'ens_final' else 0.5)
                        .assign(model=name, track=track, scenario='S1'))
            comp[name] = pf
        for name, pf in comp.items():                                   # 저장 이름: oof_test_{모델}_{트랙}_{시나리오}
            pu.save_oof(pf.assign(fold='TEST', seed=0), df, f'test_{name}', track, 'S0' if name in S0_MODELS else 'S1')
    if args.no_ensemble:
        log('일부 모델 예측 완료(앙상블·채점 파일은 전체 실행에서 작성)')
        return
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(pu.OUT / 'tables' / 'test_scores.csv', index=False, encoding='utf-8-sig')
    log(f'테스트 예측·채점 완료: {res["model"].nunique()}개 모델')


if __name__ == '__main__':
    main()
