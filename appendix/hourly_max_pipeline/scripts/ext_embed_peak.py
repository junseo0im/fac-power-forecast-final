"""추가 실험(테스트 공개 후) — 파운데이션 모델 임베딩을 더한 피크(이상) 위험 확률 모델.

Chronos-2 인코더가 예측 시점까지의 과거 전력값을 요약한 벡터(임베딩)를 피크 분류기 입력에 더해,
'사전학습 AI의 표현 + 이 공장 데이터로 학습한 분류기'가 STEP 6 분류기보다 피크를 잘 맞히는지 본다.

- 임베딩: amazon/chronos-2(d_model 768, 패치 16시간)의 Chronos2Pipeline.embed().
  출력 순서는 [문맥 패치들, REG, 출력 패치]이므로 REG = 뒤에서 두 번째. 벡터 = [REG] + 마지막 6개 문맥 패치(96시간) 평균.
  트랙 A: 대상 시각 t마다 t-1까지(직전 1344시간) / 트랙 B: 날짜마다 전날 23시까지, 그날 24시간 행이 공유.
  정전(outage)은 NaN으로 넣어 마스크한다. 추론만 하므로 결정적이다.
- 분류기·확률층: STEP 6(scripts/step6_ensemble.py)의 함수를 그대로 쓴다. 바뀌는 것은 입력뿐이다.
  (a) 기존 피처(S1) — STEP 6 재현 확인용 / (b) 임베딩 + 시각 / (c) 기존 피처 + 임베딩
  임베딩은 fold마다 학습 구간에서만 맞춘 PCA로 16차원으로 줄인다. 경로: 분류기 단독, 스태킹(분류기 확률을 각 변형으로 교체).
  isotonic 보정·F1 최대 임계값은 이전 fold로 맞춰 F2~F6에서 평가한다(STEP 6과 같음).
- 테스트(9/1~9/14)는 이미 STEP 7에서 1회 썼다. 여기의 테스트 결과는 '테스트 공개 후 추가 실험'으로 참고만 하며,
  테스트 전에 정한 규칙으로 고른 최종 모델은 바꾸지 않는다. cv_scores.csv에는 쓰지 않는다(노트북 02의 선정 재계산 보호).

실행(프로젝트 루트에서): PYTHONHASHSEED=0 N_THREADS=2 python scripts/ext_embed_peak.py [--smoke]
결과: outputs/tables/X01~X06_*.csv, outputs/preds/ext_embed_{A,B}.parquet, 임베딩 캐시 outputs/logs/ext_embed_runs/
"""
import argparse
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
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

warnings.filterwarnings('ignore')
import step6_ensemble as s6  # noqa: E402
from chronos import Chronos2Pipeline  # noqa: E402

RUN_DIR = pu.OUT / 'logs' / 'ext_embed_runs'
T = pu.OUT / 'tables'
MODEL_ID = 'amazon/chronos-2'
CTX = 1344                     # STEP 5 공변량 실험과 같은 문맥 길이
TAIL = 6                       # 마지막 6개 패치 = 96시간
N_PCA = 16
VARIANTS = {'a': '(a) 기존 피처', 'b': '(b) 임베딩 + 시각', 'c': '(c) 기존 피처 + 임베딩'}
PATHS = {'p2': '분류기', 'stack': '스태킹'}
POST_TEST = '테스트 공개 후 추가 실험(선정에 사용하지 않음)'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def origin_positions(df, track):
    """임베딩 원점(정수 위치). 원점 o의 문맥은 y[o-CTX:o] = o 직전까지. 최소 1시간의 과거가 있어야 한다."""
    pos = np.arange(1, len(df))
    return pos if track == 'A' else pos[df['hour'].to_numpy()[pos] == 0]


def make_items(y, starts):
    return [torch.tensor(y[max(0, o - CTX):o], dtype=torch.float32) for o in starts]


def pool(embs):
    """embed() 출력(변수 1개 × (패치 수 + 2) × 768) → [REG] + 마지막 TAIL개 문맥 패치 평균(1536차원)."""
    return np.stack([np.concatenate([e[0, -2].numpy(), e[0, -2 - TAIL:-2].mean(0).numpy()]) for e in embs])


def embed(pipe, y, starts, chunk=512):
    out, t0 = [], time.time()
    for i in range(0, len(starts), chunk):
        embs, _ = pipe.embed(make_items(y, starts[i:i + chunk]), batch_size=64, context_length=CTX)
        out.append(pool(embs))
        log(f'  임베딩 {min(i + chunk, len(starts))}/{len(starts)} ({time.time() - t0:.0f}초)')
    return np.vstack(out).astype(np.float32), time.time() - t0


def leak_checks(pipe, df, track, n=30, seed=0):
    """(1) 정적: 원점 문맥이 예측 시점 이전에서 끝나는지 (2) 동적: 원점 이후 값을 난수로 바꿔도 임베딩이 같은지
    (3) 배치 불변: 같은 원점을 혼자 넣어도 다른 원점과 함께 넣어도 임베딩이 같은지."""
    y = df['y_obs'].to_numpy(float)
    rng = np.random.default_rng(seed)
    pos = origin_positions(df, track)
    cand = pos[(df.index[pos] >= '2021-02-01') & (df.index[pos] < pu.TEST_START)]
    picks = np.sort(rng.choice(cand, n, replace=False))
    for o in picks:                                                           # 정적
        last_used = df.index[o - 1]
        target = df.index[o]
        assert last_used < target, f'정적 누수(트랙 {track}): {last_used} ≥ {target}'
        if track == 'B':
            assert target.hour == 0 and last_used.hour == 23, f'트랙 B 원점이 00시가 아님: {target}'
    base = pool(pipe.embed(make_items(y, picks), batch_size=64, context_length=CTX)[0])
    for k, o in enumerate(picks):                                             # 동적
        y2 = y.copy()
        y2[o:] = rng.uniform(0, 300, size=len(y2) - o)
        e2 = pool(pipe.embed(make_items(y2, [o]), batch_size=64, context_length=CTX)[0])[0]
        assert np.allclose(base[k], e2, atol=1e-4), f'동적 누수(트랙 {track}, 원점 {df.index[o]})'
    alone = pool(pipe.embed(make_items(y, picks[:1]), batch_size=64, context_length=CTX)[0])[0]
    batch_diff = float(np.abs(alone - base[0]).max())                        # 배치 불변
    assert batch_diff < 1e-3, f'배치 구성에 따라 임베딩이 달라짐: {batch_diff}'
    log(f'트랙 {track}: 누수 검사 통과(정적·동적 {n}개), 배치 불변 최대 차이 {batch_diff:.2e}')
    return n, batch_diff


def load_embeddings(pipe, df, track):
    """원점별 임베딩을 계산(또는 캐시에서 읽기)해 '행(대상 시각)별' 표로 돌려준다."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_DIR / f'emb_{track}.npz'
    starts = origin_positions(df, track)
    if path.exists():
        z = np.load(path)
        E, sec = z['E'], float(z['sec'])
        assert np.array_equal(z['starts'], starts)
        log(f'트랙 {track}: 임베딩 캐시 사용 {E.shape}')
    else:
        log(f'트랙 {track}: 임베딩 {len(starts)}개 계산')
        E, sec = embed(pipe, df['y_obs'].to_numpy(float), starts)
        np.savez_compressed(path, E=E, starts=starts, sec=sec)
    by_origin = pd.DataFrame(E, index=df.index[starts])
    if track == 'A':
        rows = by_origin.reindex(df.index)                                    # 행 t ← 원점 t(문맥은 t-1까지)
    else:
        rows = by_origin.reindex(df.index.normalize())                        # 그날 모든 행 ← 그날 00시 원점
        rows.index = df.index
    return rows, sec


def pca_features(E_rows, train_mask, prefix='emb'):
    """학습 구간 행으로만 PCA를 맞추고 전체 행을 변환한다(첫 행처럼 임베딩이 없는 행은 NaN)."""
    ok = E_rows.notna().all(axis=1).to_numpy()
    p = PCA(n_components=N_PCA, random_state=0).fit(E_rows.to_numpy()[ok & train_mask])
    Z = np.full((len(E_rows), N_PCA), np.nan)
    Z[ok] = p.transform(E_rows.to_numpy()[ok])
    return pd.DataFrame(Z, index=E_rows.index, columns=[f'{prefix}_{i:02d}' for i in range(N_PCA)])


def variant_X(X_feat, Z, df, v):
    if v == 'a':
        return X_feat
    if v == 'b':
        return Z.assign(hour=df['hour'].to_numpy())
    return X_feat.join(Z)


def eval_path(layer, s, path, lab, okt):
    raw = s['p2'].to_numpy() if path == 'p2' else layer['stack'].predict_proba(s6.stack_feats(s))[:, 1]
    pc = layer['iso'][path].predict(raw)
    t = layer['thr'][(path, '단순 피크', 'F1 최대')]
    return pc, t, s6.cls_metrics(lab[okt], pc[okt], t)


def boot_f1_diff(dates, lab, a1, a2, n_boot=2000, seed=42):
    """날짜 단위로 다시 뽑아 F1(변형) − F1(기존)의 95% 신뢰구간을 구한다."""
    def f1(l, a):
        tp, fp, fn = (l & a).sum(), (~l & a).sum(), (l & ~a).sum()
        return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    days = np.unique(dates)
    groups = {d: np.where(dates == d)[0] for d in days}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[d] for d in rng.choice(days, len(days))])
        diffs.append(f1(lab[idx], a2[idx]) - f1(lab[idx], a1[idx]))
    return np.nanpercentile(diffs, [2.5, 97.5])


def test_store(df, track, cfg, theta, p2_test):
    """STEP 7(predict_test.pred_ensembles)과 같은 방식으로 테스트 구간의 1단계 확률 재료를 다시 만든다."""
    te = df.index[(df.index >= pu.TEST_START) & (df.index <= pu.TEST_END)]

    def seed_mean(model):
        sc = 'S0' if model in ('chronos2_zs', 'mstl', 'guidebook_rf') else 'S1'
        d = pd.read_parquet(pu.OUT / 'preds' / f'oof_test_{model}_{track}_{sc}.parquet')
        g = d.groupby('timestamp').mean(numeric_only=True)
        g.index = pd.DatetimeIndex(g.index)
        return g.reindex(te)
    comp = {m: seed_mean(m) for m in cfg['candidates']}
    P = pd.DataFrame({m: comp[m]['y_pred'] for m in cfg['candidates']}, index=te)
    qsrc = [m for m in ['lgbm_q', 'chronos2_cov'] if m in comp]
    ens_q = sum(comp[m][['q10', 'q50', 'q90']] for m in qsrc) / len(qsrc)
    ens_q = pd.DataFrame(np.sort(ens_q.to_numpy(), axis=1), index=te, columns=['q10', 'q50', 'q90'])
    return {'idx': te, 'theta': theta, 'ens_q': ens_q,
            'p1q': pd.Series(pu.peak_prob_from_quantiles(ens_q, theta), index=te),
            'p2': p2_test.reindex(te), 'e_ref': P.mean(axis=1)}


def run_track(pipe, df, track, folds, seeds):
    E_rows, emb_sec = load_embeddings(pipe, df, track)
    n_leak, batch_diff = leak_checks(pipe, df, track)
    X_feat, _ = pu.make_features(df, track, 'S1')
    P, quants, info = s6.load_candidates(track)
    quants = {m: q.reindex(P.index) for m, q in quants.items()}
    stores = {v: {} for v in VARIANTS}
    for f in folds.itertuples():                                              # 1단계: fold별 확률
        theta = df.loc[df.index < f.val_start, 'y_obs'].quantile(0.95)
        base = s6.base_probs(df, P, quants, X_feat, f, theta, seeds)          # (a) = STEP 6 그대로
        stores['a'][f.fold] = base
        Z = pca_features(E_rows, (df.index < f.val_start))
        for v in ['b', 'c']:
            p2 = s6.train_classifier(df, variant_X(X_feat, Z, df, v), f, theta, True, seeds)
            stores[v][f.fold] = {**base, 'p2': p2.reindex(base['idx'])}
        log(f'트랙 {track} {f.fold}: 분류기 3종 학습 완료(θ = {theta:.0f})')
    fold_rows, pred_parts = [], []
    for fk, prior in pu.forward_folds(folds['fold']):                         # 2단계: 이전 fold로 보정해 F2~F6 평가
        s_a = stores['a'][fk]
        idx = s_a['idx']
        lab = (df.loc[idx, 'y'] >= s_a['theta']).to_numpy()
        okt = ~df.loc[idx, 'outage'].to_numpy()
        part = pd.DataFrame({'fold': fk, 'y_true': df.loc[idx, 'y'].to_numpy(), 'theta': s_a['theta'],
                             'peak': lab, 'outage': ~okt}, index=idx)
        for v in VARIANTS:
            layer = s6.fit_prob_layer(df, [stores[v][k] for k in prior])
            for path in PATHS:
                pc, t, m = eval_path(layer, stores[v][fk], path, lab, okt)
                fold_rows.append({'track': track, 'fold': fk, 'variant': VARIANTS[v], 'path': PATHS[path], **m})
                part[f'p_{v}_{path}'] = pc
                part[f'alarm_{v}_{path}'] = pc >= t
        pred_parts.append(part)
    pred = pd.concat(pred_parts)
    pred.index.name = 'timestamp'
    pred.reset_index().to_parquet(pu.OUT / 'preds' / f'ext_embed_{track}.parquet', index=False)
    # 부트스트랩: (c)·(b) − (a), 경로별
    ok = ~pred['outage'].to_numpy()
    dates = pred.index.normalize().to_numpy()[ok]
    lab = pred['peak'].to_numpy()[ok]
    boot_rows = []
    for v in ['b', 'c']:
        for path in PATHS:
            pa, pv = pred[f'p_a_{path}'].to_numpy()[ok], pred[f'p_{v}_{path}'].to_numpy()[ok]
            d = (pv - lab) ** 2 - (pa - lab) ** 2
            lo_b, hi_b = pu.block_bootstrap_ci(d)
            lo_f, hi_f = boot_f1_diff(dates, lab, pred[f'alarm_a_{path}'].to_numpy()[ok],
                                      pred[f'alarm_{v}_{path}'].to_numpy()[ok])
            boot_rows.append({'track': track, 'variant': VARIANTS[v], 'path': PATHS[path],
                              'Brier 차이(변형−기존)': d.mean(), 'Brier 차이 95% CI 하한': lo_b, 'Brier 차이 95% CI 상한': hi_b,
                              'F1 차이 95% CI 하한': lo_f, 'F1 차이 95% CI 상한': hi_f})
    # 최종 학습(8/31까지): 임베딩 gain 비율, 사후 테스트
    test_fold = next(pu.make_folds('full', include_test=True).tail(1).itertuples())
    train_mask = df.index < pu.TEST_START
    theta_t = df.loc[train_mask, 'y_obs'].quantile(0.95)
    Z_t = pca_features(E_rows, train_mask)
    import lightgbm as lgb
    Xc = variant_X(X_feat, Z_t, df, 'c')
    tr = df.index[train_mask & df['y_obs'].notna().to_numpy()]
    labt = (df.loc[tr, 'y_obs'] >= theta_t).astype(int)
    spw = (len(labt) - labt.sum()) / max(labt.sum(), 1)
    params = {k: v for k, v in s6.CLF_PARAMS.items() if k != 'tau'}
    gain = np.zeros(Xc.shape[1])
    for sd in seeds:
        mdl = lgb.LGBMClassifier(**params, scale_pos_weight=spw, subsample_freq=1, random_state=sd, n_jobs=pu.N_THREADS,
                                 deterministic=True, force_row_wise=True, verbose=-1).fit(Xc.loc[tr], labt)
        gain += mdl.booster_.feature_importance('gain')
    gain = pd.Series(gain / gain.sum(), index=Xc.columns)
    gain_row = {'track': track, '임베딩 성분 gain 비율': gain[gain.index.str.startswith('emb_')].sum(),
                '임베딩 상위 성분': gain[gain.index.str.startswith('emb_')].idxmax(),
                '전체 상위 5개 피처': ', '.join(gain.sort_values(ascending=False).index[:5])}
    cfg = pd.read_pickle(pu.OUT / 'models' / 'step6_final_config.pkl')[track]
    te = df.index[(df.index >= pu.TEST_START) & (df.index <= pu.TEST_END)]
    lab_t = (df.loc[te, 'y'] >= theta_t).to_numpy()
    ok_t = ~df.loc[te, 'outage'].to_numpy()
    test_rows = []
    for v in VARIANTS:
        Xv = X_feat if v == 'a' else variant_X(X_feat, Z_t, df, v)
        p2_t = s6.train_classifier(df, Xv, test_fold, theta_t, True, seeds)
        layer = s6.fit_prob_layer(df, list(stores[v].values()))              # 6 fold 전체로 맞춘 확률층(STEP 7과 같음)
        st = test_store(df, track, cfg, theta_t, p2_t)
        for path in PATHS:
            _, _, m = eval_path(layer, st, path, lab_t, ok_t)
            test_rows.append({'track': track, 'variant': VARIANTS[v], 'path': PATHS[path], '구분': POST_TEST, **m})
    # 시각화용 2D(테스트 이전 임베딩으로만 맞춘 PCA, 색은 θ_test 기준 피크·가동 여부)
    okE = E_rows.notna().all(axis=1).to_numpy() & train_mask
    p2d = PCA(n_components=2, random_state=0).fit(E_rows.to_numpy()[okE])
    xy = p2d.transform(E_rows.to_numpy()[okE])
    sel = df.index[okE]
    viz = pd.DataFrame({'timestamp': sel, 'track': track, 'pc1': xy[:, 0], 'pc2': xy[:, 1],
                        'op_day': df.loc[sel, 'op_day'].to_numpy(), 'hour': df.loc[sel, 'hour'].to_numpy(),
                        'peak': (df.loc[sel, 'y'] >= theta_t).to_numpy(), 'outage': df.loc[sel, 'outage'].to_numpy()})
    if track == 'B':
        viz = viz[viz['hour'] == 0]                                            # 트랙 B는 하루 1개 벡터
    info_row = {'track': track, '임베딩 수': int(E_rows.notna().all(axis=1).sum() if track == 'A' else
                                              E_rows.drop_duplicates().notna().all(axis=1).sum()),
                '임베딩 계산 시간(초)': round(emb_sec), '누수 검사 표본': n_leak, '배치 불변 최대 차이': batch_diff,
                '문맥 길이': CTX, 'PCA 차원': N_PCA, '임베딩 원래 차원': E_rows.shape[1]}
    return fold_rows, boot_rows, gain_row, test_rows, viz, info_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', nargs='*', default=['A', 'B'])
    ap.add_argument('--smoke', action='store_true', help='임베딩 256개 시간 측정과 누수 검사만 하고 끝낸다')
    args = ap.parse_args()
    pu.set_seed(42)
    df, _ = pu.load_data()
    pipe = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map='cpu')
    if args.smoke:
        for track in args.tracks:
            leak_checks(pipe, df, track)
            starts = origin_positions(df, track)[-256:]
            _, sec = embed(pipe, df['y_obs'].to_numpy(float), starts, chunk=256)
            n_all = len(origin_positions(df, track))
            log(f'트랙 {track}: 256개 {sec:.0f}초 → 전체 {n_all}개 예상 {sec / 256 * n_all / 60:.0f}분')
        return
    folds = pu.make_folds('full')
    out = {k: [] for k in ['fold', 'boot', 'gain', 'test', 'viz', 'info']}
    for track in args.tracks:
        r = run_track(pipe, df, track, folds, pu.SEEDS)
        for k, v in zip(out, r):
            out[k].extend(v if isinstance(v, list) else [v])
    folds_df = pd.DataFrame(out['fold'])
    metrics = ['f1', 'precision', 'recall', 'pr_auc', 'brier', 'ece']
    summ = folds_df.groupby(['track', 'variant', 'path'], sort=False)[metrics].mean().reset_index()
    pu.save_table(summ, 'X01_embed_peak_scores')                             # 반올림하지 않고 저장(표시할 때만 반올림)
    pu.save_table(folds_df, 'X02_embed_peak_folds')
    pu.save_table(pd.DataFrame(out['test']), 'X03_embed_peak_test')
    pu.save_table(pd.DataFrame(out['boot']), 'X04_embed_peak_bootstrap')
    pu.save_table(pd.DataFrame(out['gain']), 'X05_embed_gain')
    pu.save_table(pd.concat(out['viz']), 'X06_embed_pca2')
    pu.save_table(pd.DataFrame(out['info']), 'X07_embed_info')
    log('저장 완료: outputs/tables/X01~X07, outputs/preds/ext_embed_{A,B}.parquet')
    print(summ.round(4).to_string())


if __name__ == '__main__':
    main()
