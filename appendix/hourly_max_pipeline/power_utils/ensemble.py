"""불확실성·확률 도구(분위수 → 피크 초과 확률, 컨포멀 분위수 회귀, 경보 임계값 선택)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import QUANTILES, Q_COLS


def peak_prob_from_quantiles(q: pd.DataFrame, theta) -> np.ndarray:
    """분위수 예측으로 P(y ≥ θ)를 구한다. 분위수 점을 잇는 선형 CDF, 양 끝은 가장 바깥 분위수 간격의 2배까지 선형으로 늘인다."""
    cols = [c for c in Q_COLS if c in q and q[c].notna().any()]
    lv = np.array([QUANTILES[Q_COLS.index(c)] for c in cols])
    Q = np.sort(q[cols].to_numpy(float), axis=1)
    lo = Q[:, 0] - 2 * np.maximum(Q[:, min(1, len(cols) - 1)] - Q[:, 0], 1e-6) if len(cols) > 1 else Q[:, 0] - 1
    hi = Q[:, -1] + 2 * np.maximum(Q[:, -1] - Q[:, max(-2, -len(cols))], 1e-6) if len(cols) > 1 else Q[:, -1] + 1
    xs = np.column_stack([lo, Q, hi])
    fs = np.concatenate([[0.0], lv, [1.0]])
    th = np.broadcast_to(np.asarray(theta, float), (len(q),))
    cdf = np.array([np.interp(t, x, fs) for t, x in zip(th, xs)])
    return 1.0 - cdf


def conformal_adjust(cal: pd.DataFrame, target: pd.DataFrame, groups_cal, groups_target, alpha: float = 0.2,
                     min_n: int = 30) -> pd.DataFrame:
    """분위수 구간 [q10, q90]을 컨포멀 분위수 회귀(CQR)로 넓히거나 좁힌다.

    cal의 점수 E = max(q10 − y, y − q90)에서 그룹마다 ⌈(n+1)(1−α)⌉/n 분위수 Q를 구해 target 구간을 [q10 − Q, q90 + Q]로 바꾼다.
    그룹(예: 가동 여부 × 시간대) 표본이 min_n보다 적으면 전체 Q를 쓴다.
    """
    E = np.maximum(cal['q10'] - cal['y_true'], cal['y_true'] - cal['q90']).to_numpy()
    gc = np.asarray(groups_cal)

    def qhat(e):
        n = len(e)
        return np.quantile(e, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method='higher')

    q_all = qhat(E[~np.isnan(E)])
    table = {g: qhat(E[(gc == g) & ~np.isnan(E)]) for g in np.unique(gc) if ((gc == g) & ~np.isnan(E)).sum() >= min_n}
    adj = np.array([table.get(g, q_all) for g in np.asarray(groups_target)])
    out = target.copy()
    out['q10'] = target['q10'] - adj
    out['q90'] = target['q90'] + adj
    return out


def best_threshold(y: np.ndarray, p: np.ndarray, fn_cost: float | None = None) -> float:
    """검증 확률로 경보 임계값을 고른다. fn_cost=None이면 F1 최대, 아니면 비용 FN×fn_cost + FP×1 최소."""
    y = np.asarray(y, bool)
    best_t, best_v = 0.5, None
    for t in np.unique(np.round(np.r_[np.linspace(0.02, 0.98, 49), p], 3)):
        a = p >= t
        tp, fp, fn = (y & a).sum(), (~y & a).sum(), (y & ~a).sum()
        v = -(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0) if fn_cost is None else fn * fn_cost + fp
        if best_v is None or v < best_v:
            best_t, best_v = float(t), v
    return best_t
