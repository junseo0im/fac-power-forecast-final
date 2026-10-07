"""유의성 검정(Diebold–Mariano HLN, 블록 부트스트랩)과 테스트 전에 정한 최종 선정 규칙."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_data_cached
from .evaluation import oof_seed_mean


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 1) -> dict:
    """Diebold–Mariano 검정(Harvey–Leybourne–Newbold 소표본 보정). 두 모델 오차 e1, e2의 손실 차이 평균이 0인지 본다.

    손실 = |e|^power(기본 절대오차). h는 예측 거리(트랙 A 1, 트랙 B 24)로, 자기상관을 h−1 시차까지 반영한다.
    stat < 0 이고 p < 0.05면 모델 1이 유의하게 더 정확하다는 뜻이다.
    """
    from scipy import stats
    d = np.abs(np.asarray(e1, float)) ** power - np.abs(np.asarray(e2, float)) ** power
    d = d[~np.isnan(d)]
    n = len(d)
    dbar = d.mean()
    if np.allclose(d, 0):                      # 두 예측이 같으면 차이 없음
        return {'n': n, 'mean_loss_diff': 0.0, 'dm_hln': 0.0, 'p_value': 1.0}
    gamma = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        var = gamma[0] / n
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {'n': n, 'mean_loss_diff': dbar, 'dm_hln': stat, 'p_value': p}


def block_bootstrap_ci(d: np.ndarray, block: int = 24, n_boot: int = 2000, seed: int = 42,
                       level: float = 0.95) -> tuple[float, float]:
    """이동 블록 부트스트랩으로 평균(예: 손실 차이)의 신뢰구간을 구한다. 블록 길이 기본 24시간(하루)."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    n = len(d)
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(n - block + 1)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = (rng.choice(starts, n_blocks)[:, None] + np.arange(block)).ravel()[:n]
        means[b] = d[idx].mean()
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


SIMPLICITY = {'naive_last': 0, 'naive_week': 0, 'naive_4wk': 0, 'mstl': 1, 'arima': 1,
              'guidebook_rf': 2, 'lgbm': 2, 'lgbm_q': 2, 'lgbm_peakw': 2, 'xgb': 2, 'cat': 2,
              'nhits': 3, 'nhits_L336': 3, 'lstm': 3, 'guidebook_rnn': 3, 'chronos2_zs': 4, 'chronos2_cov': 4,
              'ens_top3': 5, 'ens_nnls': 5, 'ens_nnls_hour': 5, 'ens_final': 5}


SELECT_FOLDS = ['F2', 'F3', 'F4', 'F5', 'F6']


ALARM_ORDER = {'경로2 분류기(가중)': 0, '경로1 Chronos-2 분위수': 1, '경로1 ens_q 분위수': 1, '스태킹': 2}


def apply_selection_rule(cv: pd.DataFrame, track: str, p_equiv: float = 0.05, f1_margin: float = 0.05,
                         mae_margin: float = 0.05):
    """테스트 전에 정한 규칙으로 트랙의 점예측 모델을 고른다. 반환: (후보 표, 선정 모델 이름).

    1) 후보: S0·S1, F2~F6 다섯 fold가 모두 있는 모델(S2·TFT·Chronos-2-small 제외)
    2) 기준 1위 = F2~F6 가동일 MAE 최소
    3) 동등 = 1위와 DM(HLN) p ≥ 0.05 이고 가동일 MAE가 1위의 (1 + mae_margin)배 이내
       (10/5 보완: 검증 결과를 본 뒤 MAE 조건을 더함. 시간 단위 DM은 휴가 fold 때문에 검정력이 낮아, 훨씬 나쁜 나이브도
        '동등'이 되는 문제를 막음. 보완 전 규칙의 선정 결과는 outputs/tables/selection_result_before_rule_update.json)
    4) 동등 후보 중 단순 피크 F1이 1위보다 0.05 넘게 낮으면 제외  5) 남은 후보 중 가장 단순, 같으면 1일 추론 시간이 짧은 모델
    """
    # ens_final은 점예측이 ens_nnls와 같고 F1이 확률 경보 기준이라 점예측 후보에서 뺀다(경보는 select_alarm_path로 따로 선정)
    s = cv[(cv['track'] == track) & cv['fold'].isin(SELECT_FOLDS) & cv['scenario'].isin(['S0', 'S1'])
           & cv['model'].isin([m for m in SIMPLICITY if m != 'ens_final'])]
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)   # 추론 시간이 없는 앙상블의 빈 중앙값 경고
    nfold = s.groupby(['model', 'scenario'])['fold'].nunique()
    keys = [k for k, n in nfold.items() if n == len(SELECT_FOLDS)]
    rows = []
    for model, sc in keys:
        d = s[(s['model'] == model) & (s['scenario'] == sc)]
        rows.append({'model': model, 'scenario': sc, 'op_mae': d.loc[d['subset'] == 'op', 'mae'].mean(),
                     'mae': d.loc[d['subset'] == 'all', 'mae'].mean(), 'f1': d.loc[d['subset'] == 'all', 'f1'].mean(),
                     'infer_sec': d.loc[d['subset'] == 'all', 'infer_sec'].median(), 'simplicity': SIMPLICITY[model]})
    t = pd.DataFrame(rows).sort_values('op_mae', ignore_index=True)
    best = t.iloc[0]
    h = 1 if track == 'A' else 24
    ob = oof_seed_mean(best['model'], track, best['scenario'])
    pvals, diffs = [], []
    for r in t.itertuples():
        o = oof_seed_mean(r.model, track, r.scenario)
        j = ob.join(o[['y_pred']], rsuffix='_c', how='inner')
        j = j[j['fold'].isin(SELECT_FOLDS)]
        info = load_data_cached().loc[j.index]
        m = info['op_day'].to_numpy(bool) & ~info['outage'].to_numpy(bool)
        e_best, e_c = (j['y_true'] - j['y_pred']).to_numpy()[m], (j['y_true'] - j['y_pred_c']).to_numpy()[m]
        if r.model == best['model'] and r.scenario == best['scenario']:
            pvals.append(np.nan); diffs.append(0.0)
            continue
        res = dm_test(e_c, e_best, h=h)
        pvals.append(res['p_value']); diffs.append(res['mean_loss_diff'])
    t['DM p(1위 대비)'] = pvals
    t['가동일 절대오차 차이(후보−1위)'] = diffs
    t['DM 동등(p≥0.05)'] = t['DM p(1위 대비)'].isna() | (t['DM p(1위 대비)'] >= p_equiv)
    t['MAE 5% 이내'] = t['op_mae'] <= best['op_mae'] * (1 + mae_margin)
    t['F1 조건'] = t['f1'] >= best['f1'] - f1_margin
    t['남은 후보'] = t['DM 동등(p≥0.05)'] & t['MAE 5% 이내'] & t['F1 조건']
    pick = t[t['남은 후보']].sort_values(['simplicity', 'infer_sec']).iloc[0]
    t['선정'] = (t['model'] == pick['model']) & (t['scenario'] == pick['scenario'])
    return t, f"{pick['model']}:{pick['scenario']}"


def select_alarm_path(cls_table: pd.DataFrame, track: str, margin: float = 0.02):
    """STEP 6 확률 경로 중 F2~F6 단순 피크 F1(F1 최대 임계값)이 가장 높은 경로. 0.02 이내면 더 단순한 경로."""
    d = cls_table[(cls_table['track'] == track) & (cls_table['label'] == '단순 피크') & (cls_table['rule'] == 'F1 최대')]
    f1 = d.groupby('path')['f1'].mean().sort_values(ascending=False)
    near = [p for p in f1.index if f1[p] >= f1.iloc[0] - margin]
    return f1, min(near, key=lambda p: (ALARM_ORDER.get(p, 9), -f1[p]))
