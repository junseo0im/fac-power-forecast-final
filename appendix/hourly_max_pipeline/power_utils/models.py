"""트리·통계 모델 공통 설정(학습 가중치, Optuna 탐색 공간, LightGBM·XGBoost·CatBoost 생성, ARIMA 푸리에·외생변수)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import N_THREADS
from .features import fill_prod_plan


TAU_CHOICES = ['30', '60', 'inf']   # 최근성 가중치 시간상수 τ(일). 'inf' = 가중치 없음


PEAK_ALPHAS = [1, 2, 4]             # 피크 가중 L2의 α 후보


TREE_MODELS = {'lgbm': 'LightGBM', 'xgb': 'XGBoost', 'cat': 'CatBoost'}


def sample_weight(index: pd.DatetimeIndex, ref_time, y=None, tau='inf', alpha: float = 0.0,
                  p90: float | None = None) -> np.ndarray:
    """학습 가중치 = 최근성 가중치 exp(−경과일/τ) × 피크 가중치(1 + α·1[y ≥ 학습 p90]).

    경과일은 ref_time(검증 시작 시각)까지 남은 일수다. τ='inf'면 최근성 가중치는 모두 1이다.
    """
    w = np.ones(len(index))
    if tau != 'inf':
        age = np.asarray((pd.Timestamp(ref_time) - index) / pd.Timedelta(days=1), float)
        w = np.exp(-age / float(tau))
    if alpha:
        w = w * (1 + alpha * (np.asarray(y) >= p90))
    return w


def suggest_tree_params(trial, kind: str) -> dict:
    """Optuna 탐색 공간(모델별 하이퍼파라미터 + 최근성 τ). τ는 'tau' 키로 함께 돌려준다."""
    if kind == 'lgbm':
        p = {'n_estimators': trial.suggest_int('n_estimators', 100, 1500, step=50),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
             'num_leaves': trial.suggest_int('num_leaves', 8, 128, log=True),
             'min_child_samples': trial.suggest_int('min_child_samples', 5, 100, log=True),
             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
             'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)}
    elif kind == 'xgb':
        p = {'n_estimators': trial.suggest_int('n_estimators', 100, 1500, step=50),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
             'max_depth': trial.suggest_int('max_depth', 3, 10),
             'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 50.0, log=True),
             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
             'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)}
    elif kind == 'cat':
        p = {'iterations': trial.suggest_int('iterations', 200, 1500, step=50),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
             'depth': trial.suggest_int('depth', 4, 8),
             'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True)}
    else:
        raise ValueError(kind)
    p['tau'] = trial.suggest_categorical('tau', TAU_CHOICES)
    return p


def make_tree_model(kind: str, params: dict, seed: int, quantile: float | None = None):
    """LightGBM·XGBoost·CatBoost 회귀 모델을 같은 규칙(스레드 수·seed·결정적 설정)으로 만든다.

    params에 'tau'가 있어도 무시한다(가중치로 따로 처리). quantile을 주면 LightGBM 분위수 회귀(예: 0.1)로 만든다.
    """
    p = {k: v for k, v in params.items() if k != 'tau'}
    if kind == 'lgbm':
        import lightgbm as lgb
        obj = {'objective': 'quantile', 'alpha': quantile} if quantile is not None else {'objective': 'regression'}
        return lgb.LGBMRegressor(**p, **obj, subsample_freq=1, random_state=seed, n_jobs=N_THREADS,
                                 deterministic=True, force_row_wise=True, verbose=-1)
    assert quantile is None, '분위수 회귀는 LightGBM만 씁니다'
    if kind == 'xgb':
        import xgboost as xgb
        return xgb.XGBRegressor(**p, objective='reg:squarederror', tree_method='hist', random_state=seed,
                                n_jobs=N_THREADS)
    if kind == 'cat':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**p, loss_function='RMSE', random_seed=seed, thread_count=N_THREADS,
                                 verbose=0, allow_writing_files=False)
    raise ValueError(kind)


def fourier_terms(index: pd.DatetimeIndex, period: int, K: int, prefix: str) -> pd.DataFrame:
    """푸리에 항 sin·cos(2πk·t/period), k = 1..K. t는 2021-01-01 00시부터 지난 시간 수(달력만으로 정해짐)."""
    t = np.asarray((index - pd.Timestamp('2021-01-01')) / pd.Timedelta(hours=1), float)
    cols = {}
    for k in range(1, K + 1):
        cols[f'{prefix}_sin{k}'] = np.sin(2 * np.pi * k * t / period)
        cols[f'{prefix}_cos{k}'] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=index)


def arima_exog(df: pd.DataFrame, scenario: str, train_index) -> pd.DataFrame:
    """AutoARIMA(동적 조화 회귀) 외생변수: 일(24h)·주(168h) 푸리에 항 4개씩 + 공휴일(S0),
    + 당일 생산계획(S1, 결손은 학습 구간 중앙값), + 당일 가동 여부·비가동 후 첫 가동일(S2)."""
    X = pd.concat([fourier_terms(df.index, 24, 4, 'd'), fourier_terms(df.index, 168, 4, 'w')], axis=1)
    X['holiday'] = df['holiday'].astype(float)
    if scenario in ('S1', 'S2'):
        X['prod_day'] = df['prod_day']
        X = fill_prod_plan(X, train_index)
    if scenario == 'S2':
        X['op_today'] = df['op_day'].astype(float)
        X['after_shutdown'] = df['after_shutdown'].astype(float)
    return X
