"""STEP 6 — 앙상블·불확실성(컨포멀)·피크 위험 확률·확률보정·임계값을 검증 예측(OOF)만으로 만든다.

시간 순서를 지키는 2단계 검증: 가중치·보정기·임계값은 '이전 fold들'의 검증 예측으로 맞추고 다음 fold에 적용한다.
그래서 F2~F6(5 fold)에서만 평가하고, F1은 맞추는 데만 쓴다. 테스트(STEP 7)에는 6 fold 전체로 맞춘 설정을 한 번 적용한다.

1) 앙상블(대표 시나리오 S1, seed 평균 예측 사용)
   - ens_top3: 이전 fold MAE 상위 3개 단순평균
   - ens_nnls: 비음수 최소제곱(NNLS) 가중평균(합 1로 정규화)
   - ens_nnls_hour(트랙 B): 시간대 6구간(4시간씩)별 NNLS 가중
   - 분위수: LightGBM 분위수와 Chronos-2 공변량의 q10·q50·q90 평균(ens_q)
2) 불확실성: 컨포멀 분위수 회귀(CQR)로 [q10, q90]을 보정(목표 커버리지 0.80 ± 0.03).
   기본 = 가장 최근 이전 fold 하나로 전역 보정. 이전 fold 전체·그룹(가동 여부 × 시간대 4구간) 보정은 뒤 fold의 분포 변화를 못 따라가
   커버리지가 0.74 수준이라(검증 비교 결과) 비교용으로만 남긴다.
3) 피크 위험 확률 P(y ≥ θ), θ = fold 학습 구간 p95
   - 경로 1: 분위수 예측의 선형 CDF(Chronos-2 공변량 9개 분위수, 보정한 ens_q)
   - 경로 2: LightGBM 이진분류(scale_pos_weight, seed 5개 평균), 피처 = [단계 ③]의 트랙·S1 피처
   - 스태킹: 로지스틱 회귀([경로 1, 경로 2, (앙상블 점예측 − θ)/θ])
   - 확률보정: isotonic, 임계값: (a) F1 최대 (b) 비용가중 FN:FP = 1·3·5·10
   - 라벨: 단순 피크, 과금 피크(중간·최대부하), 과금 피크(2023 개정 시간대, 민감도)
4) 불균형 학습 비교: 가중 없음 분류 / 클래스 가중 분류 / 피크 가중 회귀(lgbm_peakw ≥ θ) / 분위수 0.9 기준 경보(lgbm_q q90 ≥ θ)

실행(프로젝트 루트에서): PYTHONHASHSEED=0 python scripts/step6_ensemble.py > outputs/logs/step6.log 2>&1
결과: cv_scores.csv(ens_top3·ens_nnls·ens_nnls_hour·ens_final, F2~F6), oof parquet,
     outputs/tables/step6_*.csv(통합표·분류·보정·불균형·혼동행렬·FN/FP 사례·가중치), outputs/models/step6_final_config.pkl(테스트용)
"""
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import power_utils as pu  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import nnls  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

warnings.filterwarnings('ignore')
T = pu.OUT / 'tables'
CANDIDATES = {
    'A': [('lgbm', 'S1'), ('lgbm_q', 'S1'), ('lgbm_peakw', 'S1'), ('xgb', 'S1'), ('cat', 'S1'), ('nhits', 'S1'),
          ('lstm', 'S1'), ('chronos2_cov', 'S1'), ('chronos2_zs', 'S0'), ('mstl', 'S0'), ('guidebook_rf', 'S0')],
    'B': [('lgbm', 'S1'), ('lgbm_q', 'S1'), ('lgbm_peakw', 'S1'), ('xgb', 'S1'), ('cat', 'S1'), ('nhits', 'S1'),
          ('lstm', 'S1'), ('chronos2_cov', 'S1'), ('chronos2_zs', 'S0'), ('mstl', 'S0'), ('arima', 'S1')],
}
FN_COSTS = [1, 3, 5, 10]
CLF_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8,
                  colsample_bytree=0.8, tau='inf')


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load_candidates(track):
    """후보 모델의 seed 평균 검증 예측을 시각 기준으로 모은다(6 fold가 모두 있는 모델만)."""
    preds, quants = {}, {}
    for model, sc in CANDIDATES[track]:
        path = pu.OUT / 'preds' / f'oof_{model}_{track}_{sc}.parquet'
        if not path.exists():
            log(f'트랙 {track}: {model} 예측 없음 → 제외')
            continue
        o = pu.oof_seed_mean(model, track, sc)
        if o['fold'].nunique() < 6:
            log(f'트랙 {track}: {model}은 {o["fold"].nunique()} fold뿐 → 앙상블 후보에서 제외')
            continue
        preds[model] = o['y_pred']
        quants[model] = o[[c for c in pu.Q_COLS if c in o]]
        base = o
    P = pd.DataFrame(preds)
    info = base[['fold', 'y_true']]
    return P, quants, info


def group_ids(df, idx, blocks=4):
    """컨포멀 그룹: 가동 여부 × 하루를 blocks개로 나눈 시간대."""
    h = df.loc[idx, 'hour'].to_numpy() // (24 // blocks)
    return df.loc[idx, 'op_day'].astype(int).to_numpy() * 10 + h


def fit_weights(P, y, ok):
    w, _ = nnls(P[ok].to_numpy(), y[ok])
    return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])


def train_classifier(df, X, fold, theta, weighted, seeds):
    """fold 학습 구간으로 피크(y ≥ θ) 이진분류기를 학습해 검증 구간 확률(seed 평균)을 돌려준다."""
    import lightgbm as lgb
    y = df['y_obs']
    tr = df.index[(df.index < fold.val_start) & y.notna()]
    va = df.index[(df.index >= fold.val_start) & (df.index <= fold.val_end)]
    lab = (y.loc[tr] >= theta).astype(int)
    spw = (len(lab) - lab.sum()) / max(lab.sum(), 1) if weighted else 1.0
    ps = []
    for s in seeds:
        p = {k: v for k, v in CLF_PARAMS.items() if k != 'tau'}
        m = lgb.LGBMClassifier(**p, scale_pos_weight=spw, subsample_freq=1, random_state=s, n_jobs=pu.N_THREADS,
                               deterministic=True, force_row_wise=True, verbose=-1)
        m.fit(X.loc[tr], lab)
        ps.append(m.predict_proba(X.loc[va])[:, 1])
    return pd.Series(np.mean(ps, axis=0), index=va)


def cls_metrics(y, p, t):
    y, a = np.asarray(y, bool), np.asarray(p) >= t
    tp, fp, fn, tn = (y & a).sum(), (~y & a).sum(), (y & ~a).sum(), (~y & ~a).sum()
    return {'f1': 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
            'precision': tp / (tp + fp) if tp + fp else np.nan, 'recall': tp / (tp + fn) if tp + fn else np.nan,
            'pr_auc': average_precision_score(y, p) if y.any() else np.nan, 'brier': float(np.mean((p - y) ** 2)),
            'ece': pu._ece(y.astype(float), np.asarray(p)), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'threshold': t}


def base_probs(df, P, quants, X, f, theta, seeds):
    """1단계(모든 fold): 맞출 것이 없는 기본 확률과 기준 점예측. 분위수 CDF·분류기는 fold 학습 구간 정보만 쓴다."""
    idx = P.index[(P.index >= f.val_start) & (P.index <= f.val_end)]
    qsrc = [m for m in ['lgbm_q', 'chronos2_cov'] if m in quants]
    ens_q = sum(quants[m].loc[idx, ['q10', 'q50', 'q90']] for m in qsrc) / len(qsrc)
    ens_q = pd.DataFrame(np.sort(ens_q.to_numpy(), axis=1), index=idx, columns=['q10', 'q50', 'q90'])
    return {'idx': idx, 'theta': theta, 'ens_q': ens_q,
            'p1': pd.Series(pu.peak_prob_from_quantiles(quants['chronos2_cov'].loc[idx], theta), index=idx)
            if 'chronos2_cov' in quants else None,
            'p1q': pd.Series(pu.peak_prob_from_quantiles(ens_q, theta), index=idx),
            'p2': train_classifier(df, X, f, theta, True, seeds),
            'p2u': train_classifier(df, X, f, theta, False, seeds),
            'e_ref': P.loc[idx].mean(axis=1)}                                # 동일 가중 평균(맞추지 않음)


PATHS = {'경로1 Chronos-2 분위수': 'p1', '경로1 ens_q 분위수': 'p1q', '경로2 분류기(가중)': 'p2', '스태킹': 'stack'}
LABELS = [('단순 피크', None), ('과금 피크', 'billing_hour'), ('과금 피크(2023 시간대)', 'billing_hour_2023')]


def stack_feats(s):
    return np.column_stack([logit(s['p1q'].to_numpy()), logit(s['p2'].to_numpy()),
                            ((s['e_ref'] - s['theta']) / s['theta']).to_numpy()])


def fit_prob_layer(df, stores):
    """여러 fold의 1단계 확률로 스태킹·isotonic 보정·임계값을 맞춘다(평가 fold 이전 fold들 또는 테스트용 6 fold 전체)."""
    idx = np.concatenate([s['idx'] for s in stores])
    yv = df.loc[idx, 'y_obs'].to_numpy()
    ok = ~np.isnan(yv)
    lab = np.concatenate([df.loc[s['idx'], 'y_obs'].to_numpy() >= s['theta'] for s in stores])
    F = np.vstack([stack_feats(s) for s in stores])
    stack = LogisticRegression(C=1.0).fit(F[ok], lab[ok])
    raw = {k: np.concatenate([s[k].to_numpy() for s in stores]) for k in ['p1', 'p1q', 'p2'] if stores[0][k] is not None}
    raw['stack'] = stack.predict_proba(F)[:, 1]
    layer = {'stack': stack, 'iso': {}, 'thr': {}}
    for name, key in PATHS.items():
        if key not in raw:
            continue
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(raw[key][ok], lab[ok])
        pc = iso.predict(raw[key][ok])
        layer['iso'][key] = iso
        for lbl, bcol in LABELS:
            b = np.ones(ok.sum(), bool) if bcol is None else df.loc[idx, bcol].to_numpy(bool)[ok]
            for rule, c in [('F1 최대', None)] + [(f'비용 FN:FP={c}:1', c) for c in FN_COSTS]:
                layer['thr'][(key, lbl, rule)] = pu.best_threshold(lab[ok] & b, pc * b, c)
    for k in ('p2u',):
        pu_raw = np.concatenate([s[k].to_numpy() for s in stores])
        layer['thr'][(k, '단순 피크', 'F1 최대')] = pu.best_threshold(lab[ok], pu_raw[ok], None)
        layer['thr'][('p2', '단순 피크', 'F1 최대(보정 전)')] = pu.best_threshold(lab[ok], raw['p2'][ok], None)
    return layer


def run_track(df, track, folds, seeds):
    P, quants, info = load_candidates(track)
    quants = {m: q.reindex(P.index) for m, q in quants.items()}
    y = info['y_true'].reindex(P.index).to_numpy()
    ok = ~df.loc[P.index, 'outage'].to_numpy()
    fold_of = info['fold'].reindex(P.index).to_numpy()
    blocks6 = df.loc[P.index, 'hour'].to_numpy() // 4
    X, _ = pu.make_features(df, track, 'S1')
    log(f'트랙 {track}: 앙상블 후보 {list(P.columns)}')
    stores = {}
    for f in folds.itertuples():                                          # 1단계: 모든 fold
        theta = df.loc[df.index < f.val_start, 'y_obs'].quantile(0.95)
        stores[f.fold] = base_probs(df, P, quants, X, f, theta, seeds)
    fdict = {f.fold: f for f in folds.itertuples()}
    out_pred = {k: [] for k in ['ens_top3', 'ens_nnls', 'ens_nnls_hour', 'ens_final']}
    cls_rows, cal_rows, imb_rows, weight_rows, cov_rows, cases = [], [], [], [], [], []
    for fk, prior in pu.forward_folds(folds['fold']):                     # 2단계: F2~F6
        f, s = fdict[fk], stores[fk]
        te, cal = fold_of == fk, np.isin(fold_of, prior) & ok
        idx_te, theta = P.index[te], s['theta']
        okt, lab_te = ok[te], y[te] >= s['theta']
        # 앙상블
        top3 = list((P[cal].sub(y[cal], axis=0)).abs().mean().sort_values().index[:3])
        w = fit_weights(P, y, cal)
        e_top3, e_nnls = P.loc[idx_te, top3].mean(axis=1), pd.Series(P[te].to_numpy() @ w, index=idx_te)
        weight_rows.append({'track': track, 'fold': fk, 'top3': ','.join(top3),
                            **{f'w_{m}': round(v, 3) for m, v in zip(P.columns, w)}})
        e_hour = e_nnls.copy()
        if track == 'B':
            for b in range(6):
                cb, tb = cal & (blocks6 == b), te & (blocks6 == b)
                e_hour[P.index[tb]] = P[tb].to_numpy() @ fit_weights(P, y, cb)
        # 컨포멀(CQR): 기본 = 가장 최근 이전 fold 하나로 전역 보정(분포가 바뀌어도 따라감, 검증에서 선택)
        last = fold_of == prior[-1]
        cal_last = last & ok
        zeros = lambda n: np.zeros(n, int)
        ens_q_cqr = pu.conformal_adjust(stores[prior[-1]]['ens_q'].assign(y_true=y[last])[ok[last]], s['ens_q'],
                                        zeros(cal_last.sum()), zeros(te.sum()))
        calq_all = pd.concat([stores[k]['ens_q'] for k in prior]).assign(y_true=y[np.isin(fold_of, prior)])
        calq_all = calq_all[ok[np.isin(fold_of, prior)]]
        intervals = [('ens_q(보정 전)', s['ens_q']), ('ens_q(CQR)', ens_q_cqr),
                     ('ens_q(CQR 비교: 이전 fold 전체·그룹 8개)',
                      pu.conformal_adjust(calq_all, s['ens_q'], group_ids(df, calq_all.index), group_ids(df, idx_te)))]
        for m in [m for m in ['lgbm_q', 'chronos2_cov'] if m in quants]:
            qm = quants[m][['q10', 'q50', 'q90']]
            intervals += [(f'{m}(보정 전)', qm.loc[idx_te]),
                          (f'{m}(CQR)', pu.conformal_adjust(qm[cal_last].assign(y_true=y[cal_last]), qm.loc[idx_te],
                                                           zeros(cal_last.sum()), zeros(te.sum())))]
        for name, q in intervals:
            inside = (y[te] >= q['q10'].to_numpy()) & (y[te] <= q['q90'].to_numpy())
            cov_rows.append({'track': track, 'fold': fk, 'interval': name, 'cov80': inside[okt].mean(),
                             'width80': (q['q90'] - q['q10']).to_numpy()[okt].mean()})
        # 확률 층(이전 fold로 맞춤)
        layer = fit_prob_layer(df, [stores[k] for k in prior])
        raw_te = {k: s[k].to_numpy() for k in ['p1', 'p1q', 'p2'] if s[k] is not None}
        raw_te['stack'] = layer['stack'].predict_proba(stack_feats(s))[:, 1]
        for name, key in PATHS.items():
            if key not in raw_te:
                continue
            pc = layer['iso'][key].predict(raw_te[key])
            cal_rows.append({'track': track, 'fold': fk, 'path': name,
                             'brier_보정전': np.mean((raw_te[key][okt] - lab_te[okt]) ** 2),
                             'ece_보정전': pu._ece(lab_te[okt].astype(float), raw_te[key][okt]),
                             'brier_보정후': np.mean((pc[okt] - lab_te[okt]) ** 2),
                             'ece_보정후': pu._ece(lab_te[okt].astype(float), pc[okt])})
            for lbl, bcol in LABELS:
                b = np.ones(te.sum(), bool) if bcol is None else df.loc[idx_te, bcol].to_numpy(bool)
                for rule in ['F1 최대'] + [f'비용 FN:FP={c}:1' for c in FN_COSTS]:
                    t = layer['thr'][(key, lbl, rule)]
                    cls_rows.append({'track': track, 'fold': fk, 'path': name, 'label': lbl, 'rule': rule,
                                     **cls_metrics((lab_te & b)[okt], (pc * b)[okt], t)})
            if key == 'stack':
                t_f1 = layer['thr'][('stack', '단순 피크', 'F1 최대')]
                pf = pd.DataFrame({'y_pred': e_nnls, 'q10': ens_q_cqr['q10'], 'q50': s['ens_q']['q50'],
                                   'q90': ens_q_cqr['q90'], 'p_peak': pc}, index=idx_te)
                out_pred['ens_final'].append((fk, pf, t_f1))
                alarm = pc >= t_f1
                cases.append(pd.DataFrame({'timestamp': idx_te, 'track': track, 'fold': fk, 'y_true': y[te],
                                           'y_pred': e_nnls.to_numpy(), 'theta': theta, 'p_raw': raw_te['stack'],
                                           'p_peak': pc, 'alarm': alarm,
                                           'peak': lab_te, 'outage': ~okt,
                                           'type': np.select([lab_te & ~alarm, ~lab_te & alarm], ['FN', 'FP'], '')}))
        # 불균형 학습 비교
        for name, key, tkey in [('가중 없음 분류', 'p2u', ('p2u', '단순 피크', 'F1 최대')),
                                ('클래스 가중 분류', 'p2', ('p2', '단순 피크', 'F1 최대(보정 전)'))]:
            imb_rows.append({'track': track, 'fold': fk, 'method': name,
                             **cls_metrics(lab_te[okt], s[key].to_numpy()[okt], layer['thr'][tkey])})
        for name, model, col in [('피크 가중 회귀(ŷ ≥ θ)', 'lgbm_peakw', None), ('L2 회귀(ŷ ≥ θ)', 'lgbm', None),
                                 ('분위수 0.9 경보(q90 ≥ θ)', 'lgbm_q', 'q90')]:
            if model not in P:
                continue
            sc = (quants[model].loc[idx_te, col] if col else P.loc[idx_te, model]).to_numpy()
            r = cls_metrics(lab_te[okt], sc[okt], theta)
            r['brier'] = r['ece'] = np.nan
            imb_rows.append({'track': track, 'fold': fk, 'method': name, **r})
        for name, e in [('ens_top3', e_top3), ('ens_nnls', e_nnls)] + ([('ens_nnls_hour', e_hour)] if track == 'B' else []):
            out_pred[name].append((fk, e.to_frame('y_pred'), None))
    # 테스트용 설정: 6 fold 전체로 맞춤(STEP 7에서 한 번만 적용)
    final = {'track': track, 'candidates': list(P.columns),
             'nnls_weights': dict(zip(P.columns, fit_weights(P, y, ok).round(4))),
             'top3': list((P[ok].sub(y[ok], axis=0)).abs().mean().sort_values().index[:3]),
             'prob_layer': fit_prob_layer(df, list(stores.values())),
             # 테스트 구간 CQR 보정 기준 = 마지막 검증 fold(F6) 하나, 전역(검증에서 고른 방식 그대로)
             'cqr_cal': stores[list(stores)[-1]]['ens_q'].assign(y_true=y[fold_of == list(stores)[-1]])[ok[fold_of == list(stores)[-1]]],
             'clf_params': CLF_PARAMS}
    if track == 'B':
        final['nnls_hour_weights'] = {b: dict(zip(P.columns, fit_weights(P, y, ok & (blocks6 == b)).round(4)))
                                      for b in range(6)}
    return out_pred, cls_rows, cal_rows, imb_rows, weight_rows, cov_rows, cases, final


def main():
    seeds = pu.SEEDS
    df, _ = pu.load_data()
    folds = pu.make_folds('full')
    all_scores, finals = [], {}
    tabs = {k: [] for k in ['cls', 'cal', 'imb', 'weights', 'cov', 'cases']}
    for track in ['A', 'B']:
        out_pred, cls_rows, cal_rows, imb_rows, weight_rows, cov_rows, cases, final = run_track(df, track, folds, seeds)
        finals[track] = final
        for name, items in out_pred.items():
            if not items:
                continue
            frames = []
            for fk, pf, t in items:
                f = next(x for x in folds.itertuples() if x.fold == fk)
                sc = pu.evaluate(pf, df, df.index[df.index < f.val_start], threshold=t if t is not None else 0.5)
                all_scores.append(sc.assign(model=name, track=track, scenario='S1', fold=fk, seed=0, deterministic=True))
                frames.append(pf.assign(fold=fk, seed=0))
            pu.save_oof(pd.concat(frames), df, name, track, 'S1')
        tabs['cls'] += cls_rows; tabs['cal'] += cal_rows; tabs['imb'] += imb_rows
        tabs['weights'] += weight_rows; tabs['cov'] += cov_rows; tabs['cases'] += cases
        log(f'트랙 {track} 완료')
    pu.upsert_cv_scores(pd.concat(all_scores, ignore_index=True))
    for k, name in [('cls', 'step6_classification'), ('cal', 'step6_calibration'), ('imb', 'step6_imbalance'),
                    ('weights', 'step6_ensemble_weights'), ('cov', 'step6_coverage')]:
        pd.DataFrame(tabs[k]).to_csv(T / f'{name}.csv', index=False, encoding='utf-8-sig')
    cases = pd.concat(tabs['cases'], ignore_index=True)
    cond = df[['hour', 'dow', 'tou_load', 'op_day', 'prod_bin', 'tou_season', 'after_shutdown', 'billing_hour']]
    cases = cases.join(cond, on='timestamp')
    cases.to_csv(T / 'step6_alarm_cases.csv', index=False, encoding='utf-8-sig')
    cases[cases['type'] != ''].to_csv(T / 'step6_fn_fp_cases.csv', index=False, encoding='utf-8-sig')
    pd.to_pickle(finals, pu.OUT / 'models' / 'step6_final_config.pkl')
    summary = {tr: {'candidates': fin['candidates'], 'nnls_weights': fin['nnls_weights'], 'top3': fin['top3'],
                    'thresholds': {' | '.join(k): round(v, 3) for k, v in fin['prob_layer']['thr'].items()},
                    **({'nnls_hour_weights': fin['nnls_hour_weights']} if 'nnls_hour_weights' in fin else {})}
               for tr, fin in finals.items()}
    (T / 'step6_final_config.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    log('STEP 6 완료: cv_scores(ens_*), step6_*.csv, step6_final_config 저장')


if __name__ == '__main__':
    main()
