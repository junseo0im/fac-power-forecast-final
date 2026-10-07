"""나이브 기준선 3종(직전값, 1주 전 같은 시각, 최근 4주 같은 요일·시각 평균)."""
from __future__ import annotations

import pandas as pd

from .config import BASELINES, TEST_START
from .folds import fold_index
from .evaluation import CV_COLS, evaluate, save_oof, upsert_cv_scores


def baseline_predict(df: pd.DataFrame, name: str, track: str) -> pd.Series:
    """나이브 기준선 예측값(모든 시각)을 돌려준다. outage 값(NaN)은 건너뛴다.

    - naive_last: 트랙 A는 직전 시각 값, 트랙 B는 전날 23시 값을 그날 24시간에 그대로 씀
    - naive_week: 1주(168시간) 전 같은 시각, 비어 있으면 2주 전 값
    - naive_4wk: 최근 4주 같은 요일·시각 평균
    """
    y = df['y_obs']
    if name == 'naive_last':
        s = y.ffill().shift(1)
        if track == 'B':
            s = s.where(df['hour'] == 0).groupby(df['date']).transform('first')
    elif name == 'naive_week':
        s = y.shift(168).fillna(y.shift(336))
    elif name == 'naive_4wk':
        s = pd.concat([y.shift(168 * k) for k in range(1, 5)], axis=1).mean(axis=1)
    else:
        raise ValueError(name)
    return s.rename('y_pred')


def run_baselines(df: pd.DataFrame, folds: pd.DataFrame, tracks=('A', 'B')) -> pd.DataFrame:
    """기준선 3종을 검증 fold마다 평가해 cv_scores.csv와 oof parquet에 저장하고 점수표를 돌려준다."""
    assert not (folds['fold'] == 'TEST').any(), '기준선 단계에서는 테스트 구간을 평가하지 않습니다'
    all_rows = []
    for name in BASELINES:
        for track in tracks:
            full = baseline_predict(df, name, track)
            rows, preds = [], []
            for _, f in folds.iterrows():
                tr, va = fold_index(df.index, f)
                assert va.max() < TEST_START
                p = full.loc[va].to_frame()
                sc = evaluate(p, df, tr)
                sc.insert(0, 'fold', f['fold'])
                rows.append(sc)
                preds.append(p.assign(fold=f['fold'], seed=0))
            res = pd.concat(rows).assign(model=name, track=track, scenario='S0', seed=0, deterministic=True)
            save_oof(pd.concat(preds), df, name, track, 'S0')
            all_rows.append(res)
    scores = pd.concat(all_rows, ignore_index=True)
    upsert_cv_scores(scores)
    return scores.reindex(columns=CV_COLS)
