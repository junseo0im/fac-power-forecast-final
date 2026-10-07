"""평가 지표(회귀·확률·분류)와 결과 저장·불러오기(cv_scores.csv, 검증 예측 parquet, 요약표)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CV_PATH, OUT, QUANTILES, Q_COLS


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """기대 보정 오차(ECE): 확률을 10구간으로 나눠 |실제 비율 − 평균 확률|을 표본 수로 가중평균."""
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def _cls_metrics(label: np.ndarray, pred_label: np.ndarray, score: np.ndarray,
                 prob: np.ndarray | None, suffix: str = '') -> dict:
    from sklearn.metrics import average_precision_score
    tp = int((label & pred_label).sum()); fp = int((~label & pred_label).sum())
    fn = int((label & ~pred_label).sum()); tn = int((~label & ~pred_label).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    out = {f'f1{suffix}': f1, f'precision{suffix}': prec, f'recall{suffix}': rec}
    if not suffix:
        out.update({'pr_auc': average_precision_score(label, score) if label.any() else np.nan,
                    'brier': float(np.mean((prob - label) ** 2)) if prob is not None else np.nan,
                    'ece': _ece(label.astype(float), prob) if prob is not None else np.nan,
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})
    return out


def evaluate(pred: pd.DataFrame, df: pd.DataFrame, train_index, theta_q: float = 0.95,
             threshold: float = 0.5, bill_col: str = 'billing_hour') -> pd.DataFrame:
    """회귀·확률·분류 지표를 전체(all)·가동일(op)·피크시간(peak)별로 한 번에 계산한다.

    pred: 인덱스가 시각(ts)이고 y_pred 열(필수), q10~q90·p_peak 열(선택)이 있는 표
    train_index: 이 fold의 학습 구간 인덱스. θ(피크 임계값), 학습 p90, MASE 분모를 여기서만 계산한다.
    - outage 시간은 평가에서 뺀다. 피크 라벨 = y ≥ θ(학습 구간 y의 theta_q 분위수)
    - pinball은 있는 분위수 열의 평균(예: q10·q50·q90만 있으면 그 3개), 80% 구간은 q10·q90이 있을 때 계산
    - p_peak가 없으면 점예측이 θ 이상인지로 경보를 정하고(F1 등), PR-AUC 점수로 y_pred를 쓴다. Brier·ECE는 비운다.
    - 과금 피크(_bill) = 피크이면서 중간·최대부하 시간. bill_col='billing_hour_2023'이면 2023 개정 시간대 기준
    """
    y_tr = df.loc[train_index, 'y_obs']
    theta, p90 = y_tr.quantile(theta_q), y_tr.quantile(0.9)
    yo = df['y_obs']
    mase_scale = (yo - yo.shift(168)).abs().loc[train_index].mean()

    d = pred.join(df[['y', 'outage', 'op_day', 'date', 'hour']], how='left')
    d['billing_hour'] = df.loc[d.index, bill_col].to_numpy()
    d = d[~d['outage'] & d['y_pred'].notna()]
    q_avail = [c for c in Q_COLS if c in d and d[c].notna().any()]      # 일부 분위수(예: q10·q50·q90)만 있어도 계산
    q_levels = np.array([QUANTILES[Q_COLS.index(c)] for c in q_avail])
    has_p = 'p_peak' in d and d['p_peak'].notna().any()
    rows = []
    for subset, mask in [('all', np.ones(len(d), bool)), ('op', d['op_day'].to_numpy(bool)),
                         ('peak', (d['y'] >= p90).to_numpy())]:
        s = d[mask]
        if s.empty:
            continue
        y, yp = s['y'].to_numpy(), s['y_pred'].to_numpy()
        err = np.abs(y - yp)
        r = {'subset': subset, 'n': len(s), 'theta_q': theta_q, 'theta': theta,
             'mae': err.mean(), 'rmse': np.sqrt(((y - yp) ** 2).mean()), 'mase': err.mean() / mase_scale,
             'peak_mae': err[y >= p90].mean() if (y >= p90).any() else np.nan}
        if subset != 'peak':
            g = s.groupby('date')
            r['daily_max_err'] = (g['y'].max() - g['y_pred'].max()).abs().mean()
            h_true = s.loc[g['y'].idxmax(), 'hour'].to_numpy()
            h_pred = s.loc[g['y_pred'].idxmax(), 'hour'].to_numpy()
            r['peak_hour_hit'] = np.mean(np.abs(h_true - h_pred) <= 1)
        if q_avail:
            diff = y[:, None] - s[q_avail].to_numpy()
            r['pinball'] = np.mean(np.maximum(q_levels * diff, (q_levels - 1) * diff))
            if {'q10', 'q90'} <= set(q_avail):
                r['cov80'] = np.mean((y >= s['q10']) & (y <= s['q90']))
                r['width80'] = np.mean(s['q90'] - s['q10'])
        if subset != 'peak':
            label = y >= theta
            r['n_pos'] = int(label.sum())
            prob = s['p_peak'].to_numpy() if has_p else None
            pred_label = prob >= threshold if has_p else yp >= theta
            r.update(_cls_metrics(label, pred_label, prob if has_p else yp, prob))
            bill = s['billing_hour'].to_numpy(bool)
            r.update(_cls_metrics(label & bill, pred_label & bill, yp, None, suffix='_bill'))
        rows.append(r)
    return pd.DataFrame(rows)


CV_COLS = ['model', 'track', 'scenario', 'fold', 'seed', 'deterministic', 'subset', 'n', 'n_pos', 'theta_q',
           'theta', 'mae', 'rmse', 'mase', 'peak_mae', 'daily_max_err', 'peak_hour_hit', 'pinball', 'cov80',
           'width80', 'f1', 'precision', 'recall', 'pr_auc', 'brier', 'ece', 'tp', 'fp', 'fn', 'tn',
           'f1_bill', 'precision_bill', 'recall_bill', 'train_sec', 'infer_sec']


def upsert_cv_scores(rows: pd.DataFrame, path: Path = CV_PATH) -> pd.DataFrame:
    """cv_scores.csv에 결과를 추가한다. 같은 (model, track, scenario) 결과가 있으면 새 결과로 바꾼다."""
    rows = rows.reindex(columns=CV_COLS)
    if path.exists():
        old = pd.read_csv(path, encoding='utf-8-sig')
        keys = set(map(tuple, rows[['model', 'track', 'scenario']].drop_duplicates().to_numpy()))
        old = old[[k not in keys for k in map(tuple, old[['model', 'track', 'scenario']].to_numpy())]]
        rows = pd.concat([old, rows], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False, encoding='utf-8-sig')
    return rows


def save_oof(preds: pd.DataFrame, df: pd.DataFrame, model: str, track: str, scenario: str) -> Path:
    """검증 예측을 조건 컬럼과 함께 outputs/preds/oof_{model}_{track}_{scenario}.parquet로 저장한다.

    preds: 인덱스 ts, 열 fold·seed·y_pred(+ q10~q90, p_peak)
    """
    out = preds.copy()
    for c in Q_COLS + ['p_peak']:
        if c not in out:
            out[c] = np.nan
    info = df.loc[out.index]
    out.insert(0, 'timestamp', out.index)
    out['y_true'] = info['y'].to_numpy()
    cond = {'hour': 'hour', 'dow': 'dow', 'tou': 'tou_load', 'op_day': 'op_day', 'prod_bin': 'prod_bin',
            'season': 'tou_season', 'after_shutdown': 'after_shutdown', 'outage': 'outage'}
    for new, src in cond.items():
        out[new] = info[src].to_numpy()
    cols = ['timestamp', 'fold', 'seed', 'y_true', 'y_pred'] + Q_COLS + ['p_peak'] + list(cond)
    path = OUT / 'preds' / f'oof_{model}_{track}_{scenario}.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    out[cols].reset_index(drop=True).to_parquet(path, index=False)
    return path


def load_oof(model: str, track: str, scenario: str = 'S0') -> pd.DataFrame:
    """저장된 검증 예측(outputs/preds/oof_{model}_{track}_{scenario}.parquet)을 불러온다."""
    return pd.read_parquet(OUT / 'preds' / f'oof_{model}_{track}_{scenario}.parquet')


def mae_by(model: str, track: str, by: str = 'hour', scenario: str = 'S0') -> pd.Series:
    """검증 예측의 MAE를 조건 컬럼(by: hour·dow·tou·op_day 등)별로 계산한다(outage 제외, fold·seed 전체 평균)."""
    p = load_oof(model, track, scenario)
    p = p[~p['outage']]
    return (p['y_true'] - p['y_pred']).abs().groupby(p[by]).mean()


def summarize_scores(cv: pd.DataFrame, subset: str = 'all',
                     metrics=('mae', 'mase', 'peak_mae', 'f1', 'pr_auc')) -> pd.DataFrame:
    """fold·seed 결과를 모델×트랙×시나리오별 '평균 ± 표준편차'와 중앙값으로 요약한다."""
    s = cv[cv['subset'] == subset]
    g = s.groupby(['model', 'track', 'scenario'])
    out = pd.DataFrame(index=g.size().index)
    for m in metrics:
        out[f'{m} 평균±표준편차'] = g[m].mean().round(3).astype(str) + ' ± ' + g[m].std().round(3).astype(str)
        out[f'{m} 중앙값'] = g[m].median().round(3)
    out['fold 수'] = g['fold'].nunique()
    return out.reset_index()


def oof_seed_mean(model: str, track: str, scenario: str = 'S0') -> pd.DataFrame:
    """검증 예측을 시각별 seed 평균으로 묶는다(점예측·분위수 모두 평균). 최종 예측은 seed 평균이라는 규칙을 따른다."""
    p = load_oof(model, track, scenario)
    num = ['y_true', 'y_pred'] + [c for c in Q_COLS + ['p_peak'] if p[c].notna().any()]
    out = p.groupby('timestamp').agg({**{c: 'mean' for c in num}, 'fold': 'first'})
    out.index = pd.DatetimeIndex(out.index, name='ts')
    return out
