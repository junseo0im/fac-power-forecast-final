"""STEP 2 — 가이드북 RF를 시간 단위 트랙 A로 옮겨 (a) 무작위 70:30 분할과 (b) 시간 fold로 평가한다.

가이드북과 같은 것: RandomForestRegressor(max_depth=20, n_estimators=100), 피처 구성(시각·기상·요일·일·월 + 현재 값)
가이드북과 다른 것:
- 단위: 15분 → 1시간. 직전 시각(t-1) 행의 피처 + y(t-1)로 대상 시각 y(t)를 예측(power_utils.guidebook_rf_features)
- 타깃: 다음 15분 값 → 시간 피크 y. outage 시간은 학습·평가에서 뺀다
- 분할: (a) 가이드북처럼 무작위 70:30(테스트 구간 9/1~9/14는 제외한 나머지에서) / (b) 확장 윈도우 검증 6 fold
- seed: 트리 모델 규칙대로 SEEDS 5개(무작위 분할은 분할 seed도 같이 바꿈)

실행(프로젝트 루트에서): PYTHONHASHSEED=0 python scripts/train_guidebook_rf.py > outputs/logs/guidebook_rf.log 2>&1
결과
- outputs/tables/cv_scores.csv               : model='guidebook_rf'(시간 fold만, 트랙 A·S0)
- outputs/preds/oof_guidebook_rf_A_S0.parquet
- outputs/tables/guidebook_rf_split_scores.csv: 무작위 분할과 시간 fold 점수를 한 표에(split 열로 구분, 학습 MAE 포함).
  무작위 분할은 평가 기간이 1~8월 전체라 시간 fold(6~8월)와 기간이 다르므로, 같은 기간 행만 다시 채점한 결과도 함께 둔다.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import power_utils as pu  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

MODEL = 'guidebook_rf'


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def mean_mae(score_frames) -> float:
    """점수표 여러 개의 전체(all) MAE 평균(로그 출력용)."""
    return float(np.mean([s.loc[s['subset'] == 'all', 'mae'].iloc[0] for s in score_frames]))


def fit_eval(df, X, y, fit_idx, eval_idx, theta_idx, seed):
    """RF를 fit_idx로 학습하고 eval_idx를 예측해 점수·예측·학습 MAE를 돌려준다. θ·MASE 분모는 theta_idx로 계산."""
    rf = RandomForestRegressor(max_depth=20, random_state=seed, n_jobs=pu.N_THREADS)   # 가이드북 [코드 75]
    t0 = time.time()
    rf.fit(X.loc[fit_idx], y.loc[fit_idx])
    train_sec = time.time() - t0
    t0 = time.time()
    p = rf.predict(X.loc[eval_idx])
    infer_sec = (time.time() - t0) / (len(eval_idx) / 24)       # 1일(24시간)치 예측 시간
    train_mae = float(np.mean(np.abs(rf.predict(X.loc[fit_idx]) - y.loc[fit_idx])))
    pf = pd.DataFrame({'y_pred': p}, index=eval_idx)
    sc = pu.evaluate(pf, df, theta_idx).assign(train_sec=train_sec, infer_sec=infer_sec, train_mae=train_mae)
    return sc, pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fast', 'full'], default='full')
    ap.add_argument('--n-seeds', type=int, default=5)
    args = ap.parse_args()

    df, _ = pu.load_data()
    X, _ = pu.guidebook_rf_features(df)
    y = df['y_obs']
    usable = X.notna().all(axis=1) & y.notna() & (df.index < pu.TEST_START)   # 테스트 구간은 어디에도 안 씀
    seeds = pu.SEEDS[:args.n_seeds]
    folds = pu.make_folds(args.mode)
    log(f'시작: 사용 가능 {int(usable.sum()):,}시간, fold {list(folds.fold)}, seed {seeds}, 스레드 {pu.N_THREADS}')

    rows, preds = [], []
    for fold in folds.itertuples():                                           # (b) 시간 fold
        fit_idx = df.index[usable & (df.index < fold.val_start)]
        eval_idx = df.index[(df.index >= fold.val_start) & (df.index <= fold.val_end) & X.notna().all(axis=1)]
        theta_idx = df.index[df.index < fold.val_start]
        for seed in seeds:
            sc, pf = fit_eval(df, X, y, fit_idx, eval_idx, theta_idx, seed)
            rows.append(sc.assign(split='시간 fold', fold=fold.fold, seed=seed))
            preds.append(pf.assign(fold=fold.fold, seed=seed))
        log(f'{fold.fold}: seed 평균 MAE {mean_mae(rows[-len(seeds):]):.2f}')

    pool = df.index[usable]                                                   # (a) 무작위 70:30
    val_lo, val_hi = folds['val_start'].min(), folds['val_end'].max()
    for seed in seeds:
        fit_idx, eval_idx = train_test_split(pool, test_size=0.3, random_state=seed)
        fit_idx, eval_idx = fit_idx.sort_values(), eval_idx.sort_values()
        sc, pf = fit_eval(df, X, y, fit_idx, eval_idx, fit_idx, seed)
        rows.append(sc.assign(split='무작위 70:30', fold='RANDOM', seed=seed))
        # 평가 기간 차이를 없앤 비교: 무작위 평가 행 중 시간 fold와 같은 기간(검증 6 fold 구간)에 속한 행만
        same = pf[(pf.index >= val_lo) & (pf.index <= val_hi)]
        sc2 = pu.evaluate(same, df, fit_idx).assign(train_sec=sc['train_sec'].iloc[0], infer_sec=np.nan,
                                                     train_mae=sc['train_mae'].iloc[0])
        rows.append(sc2.assign(split='무작위 70:30(검증 기간 행만)', fold='RANDOM', seed=seed))
    log(f'무작위 70:30: seed 평균 MAE {mean_mae(rows[-2 * len(seeds)::2]):.2f}, '
        f'그중 검증 기간 행만 {mean_mae(rows[-2 * len(seeds) + 1::2]):.2f}')

    allsc = pd.concat(rows, ignore_index=True)
    allsc.to_csv(pu.OUT / 'tables' / 'guidebook_rf_split_scores.csv', index=False, encoding='utf-8-sig')
    cv = allsc[allsc['split'] == '시간 fold'].assign(model=MODEL, track='A', scenario='S0', deterministic=False)
    pu.upsert_cv_scores(cv)
    pu.save_oof(pd.concat(preds), df, MODEL, 'A', 'S0')
    log('완료: cv_scores.csv, oof_guidebook_rf_A_S0.parquet, guidebook_rf_split_scores.csv 저장')


if __name__ == '__main__':
    main()
