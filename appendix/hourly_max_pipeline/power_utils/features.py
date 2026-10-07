"""트랙·시나리오별 피처 생성과 누수 검사(피처 가용성 표, 정적 assert, 동적 교란 검사)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import QUARTER_COLS, TEST_START, WEATHER_COLS


GROUP_CAL, GROUP_LAG, GROUP_S1, GROUP_S2, GROUP_WX = '달력', '과거값', 'S1 생산계획', 'S2 가동 오라클', '기상(완벽 예보 가정)'


KIND_LABEL = {'lag_h': '과거 시차(시간)', 'lag_d': '전날 이전 일 집계', 'known': '미리 아는 값'}


def make_features(df: pd.DataFrame, track: str, scenario: str = 'S1',
                  weather: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """트랙·시나리오에서 허용되는 피처만 만들고 (X, 피처 가용성 표)를 돌려준다.

    - 트랙 A(매 정시 → 다음 1시간): 직전 시각까지의 값, 즉 1시간 이상 지난 값만 쓴다.
    - 트랙 B(매일 00:00 → 그날 24시간): 전날 23:59까지의 값만 쓴다. 시간 시차는 24 이상이어야 한다.
    - 시나리오: S0 달력·과거값, S1 + 당일 생산계획, S2 + 당일 가동 여부(오라클). weather=True면 기상 실측을 '완벽 예보'로 넣는다.
    시차 규칙을 어기는 피처가 있거나 시나리오에 없는 '미리 아는 값'이 들어오면 AssertionError로 멈춘다.
    """
    assert track in ('A', 'B') and scenario in ('S0', 'S1', 'S2')
    y = df['y_obs']
    cols, spec = {}, []

    def add(name, values, group, kind, lag, desc):
        cols[name] = values
        spec.append({'피처': name, '그룹': group, 'kind': kind, '최소 시차': lag, '설명': desc})

    for name, col, desc in [('hour', 'hour', '대상 시각(0~23)'), ('dow', 'dow', '요일(0=월)'),
                            ('month', 'month', '월'), ('weekend', 'weekend', '토·일'),
                            ('holiday', 'holiday', '공휴일·대체공휴일'), ('shift', 'shift', '주간 교대(09~17시)'),
                            ('tou_season', 'tou_season_code', '한전 계절(0 겨울·1 봄가을·2 여름)'),
                            ('tou_load', 'tou_load_code', '한전 시간대(0 경부하·1 중간·2 최대)'),
                            ('billing_hour', 'billing_hour', '기본요금 반영 시간(중간·최대부하)')]:
        add(name, df[col], GROUP_CAL, 'known', None, desc)

    if track == 'A':
        for k in (1, 2, 3, 24, 168):
            add(f'y_lag{k}', y.shift(k), GROUP_LAG, 'lag_h', k, f'{k}시간 전 시간 피크')
        qprev = df[QUARTER_COLS].where(~df['outage'], np.nan).shift(1)
        add('q4_lag1', qprev['60분'], GROUP_LAG, 'lag_h', 1, '직전 시간 마지막 15분 값')
        add('q_min_lag1', qprev.min(axis=1), GROUP_LAG, 'lag_h', 1, '직전 시간 15분 값 4개의 최소')
        add('q_slope_lag1', qprev.to_numpy() @ np.array([-1.5, -0.5, 0.5, 1.5]) / 5.0,
            GROUP_LAG, 'lag_h', 1, '직전 시간 15분 값 4개의 기울기(kW/15분)')
        for k in (3, 6, 24):
            add(f'y_ma{k}', y.shift(1).rolling(k, min_periods=1).mean(), GROUP_LAG, 'lag_h', 1,
                f'직전 {k}시간 평균')
    else:
        for k, d in ((24, 1), (168, 7), (336, 14)):
            add(f'y_d{d}', y.shift(k), GROUP_LAG, 'lag_h', k, f'{d}일 전 같은 시각')
        add('y_dow4w', pd.concat([y.shift(168 * j) for j in range(1, 5)], axis=1).mean(axis=1),
            GROUP_LAG, 'lag_h', 168, '최근 4주 같은 요일·시각 평균')
        daily = pd.DataFrame({'dmax': y.groupby(df['date']).max(),
                              'dmean': df['usage'].where(~df['outage']).groupby(df['date']).mean(),
                              'op': df.groupby('date')['op_day'].first().astype(float)})
        prev = daily.shift(1)
        for name, src, desc in [('prev_day_max', prev['dmax'], '전날 최대 시간 피크'),
                                ('prev_day_mean', prev['dmean'], '전날 평균 사용량'),
                                ('prev_day_op', prev['op'], '전날 가동 여부'),
                                ('op_days_7d', daily['op'].shift(1).rolling(7, min_periods=1).sum(), '최근 7일 가동일 수')]:
            add(name, df['date'].map(src).to_numpy(), GROUP_LAG, 'lag_d', 1, desc)
        add('op_lastweek', df['date'].map(daily['op'].shift(7)).to_numpy(), GROUP_LAG, 'lag_d', 7,
            '1주 전 같은 요일 가동 여부')

    if scenario in ('S1', 'S2'):
        add('prod_day', df['prod_day'], GROUP_S1, 'known', None, '당일 생산량 일합계(전일 확정 가정, 결손일 NaN)')
        add('prod_missing', df['prod_missing'], GROUP_S1, 'known', None, '생산량 결손일 표시')
    if scenario == 'S2':
        add('op_today', df['op_day'], GROUP_S2, 'known', None, '당일 주간 가동 여부(실제 전력에서 역산)')
        add('after_shutdown', df['after_shutdown'], GROUP_S2, 'known', None, '비가동일 다음 첫 가동일')
    if weather:
        for c in WEATHER_COLS:
            add(c, df[c], GROUP_WX, 'known', None, f'대상 시각 {c} 실측')

    X = pd.DataFrame(cols, index=df.index).astype(float)
    avail = pd.DataFrame(spec)
    avail.insert(0, '트랙', track)
    avail.insert(1, '시나리오', scenario + ('+기상' if weather else ''))
    assert_feature_availability(avail, track, scenario, weather)
    avail['종류'] = avail['kind'].map(KIND_LABEL)
    avail['예측 시점에 알 수 있는 이유'] = avail.apply(lambda r: _avail_reason(r, track), axis=1)
    return X, avail


def assert_feature_availability(avail: pd.DataFrame, track: str, scenario: str, weather: bool = False) -> None:
    """피처 가용성 표를 검사한다(정적 누수 검사). 규칙을 어기면 AssertionError로 멈춘다.

    - 시간 시차: 트랙 A는 1 이상, 트랙 B는 24 이상(lag<24 금지)
    - 일 집계: 1일 전 이전 하루만
    - 미리 아는 값: 달력은 항상, 생산계획은 S1·S2, 가동 여부는 S2, 기상은 weather=True일 때만
    """
    min_h = 24 if track == 'B' else 1
    lag = avail['최소 시차'].astype(float)
    bad = avail[((avail['kind'] == 'lag_h') & (lag < min_h)) | ((avail['kind'] == 'lag_d') & (lag < 1))]
    assert bad.empty, f'트랙 {track}에서 예측 시점에 알 수 없는 피처: {bad["피처"].tolist()}'
    allowed = {GROUP_CAL} | ({GROUP_S1} if scenario != 'S0' else set()) \
        | ({GROUP_S2} if scenario == 'S2' else set()) | ({GROUP_WX} if weather else set())
    extra = set(avail.loc[avail['kind'] == 'known', '그룹']) - allowed
    assert not extra, f'시나리오 {scenario}에 허용되지 않은 정보: {extra}'


def _avail_reason(row, track: str) -> str:
    if row['kind'] == 'lag_h':
        cut = '00시 기준 전날 23시까지' if track == 'B' else '정시 기준 직전 시간까지'
        return f'{int(row["최소 시차"])}시간 전 값 → {cut}의 값만 사용'
    if row['kind'] == 'lag_d':
        return f'{int(row["최소 시차"])}일 전 이전 하루 집계 → 그날 00시에 이미 확정'
    return {GROUP_CAL: '달력·요금표로 미리 앎', GROUP_S1: '전날 확정된 생산계획이라고 가정',
            GROUP_S2: '가동 캘린더를 공유한다고 가정(상한 실험용)',
            GROUP_WX: '실측 = 완벽 예보라고 가정(추가 실험용)'}[row['그룹']]


def check_no_leak(df: pd.DataFrame, track: str, scenario: str = 'S1', weather: bool = False,
                  n_checks: int = 30, seed: int = 0, feature_fn=None) -> int:
    """예측 시점 이후 값을 무작위로 바꿔도 과거값 피처가 그대로인지 확인한다(동적 누수 검사).

    예측 시점(트랙 A: 대상 정시, 트랙 B: 대상일 00시) 이후의 전력값·기상·가동 여부를 난수로 바꾼 뒤
    피처를 다시 만들어, 대상 행의 과거값 피처가 원래와 같으면 통과한다. '미리 아는 값'은 가정으로 허용하므로 비교에서 뺀다.
    feature_fn(df) → (X, 가용성 표)를 주면 make_features 대신 그 함수를 검사한다(예: guidebook_rf_features).
    """
    if feature_fn is None:
        feature_fn = lambda d: make_features(d, track, scenario, weather)
    X_full, avail = feature_fn(df)
    lag_cols = avail.loc[avail['kind'] != 'known', '피처'].tolist()
    rng = np.random.default_rng(seed)
    cand = df.index[(df.index >= '2021-02-01') & (df.index < TEST_START)]
    if track == 'B':
        cand = cand[cand.hour == 0]
    for origin in pd.DatetimeIndex(rng.choice(cand, n_checks, replace=False)):
        d2 = df.copy()
        fut = d2.index >= origin
        noise = rng.uniform(0, 300, size=(fut.sum(), 1))
        for c in ['y', 'y_obs', 'usage'] + QUARTER_COLS + WEATHER_COLS:
            d2.loc[fut, c] = noise[:, 0]
        d2.loc[fut, 'op_day'] = ~d2.loc[fut, 'op_day'].astype(bool)
        Xp, _ = feature_fn(d2)
        rows = [origin] if track == 'A' else df.index[df['date'] == origin.normalize()]
        a, b = X_full.loc[rows, lag_cols].to_numpy(), Xp.loc[rows, lag_cols].to_numpy()
        same = np.isclose(a, b, equal_nan=True).all(axis=0)
        assert same.all(), f'누수 발견(트랙 {track}, 시점 {origin}): {np.array(lag_cols)[~same].tolist()}'
    return n_checks


def fill_prod_plan(X: pd.DataFrame, train_index) -> pd.DataFrame:
    """생산계획 결손(NaN)을 학습 구간 생산일의 중앙값으로 채운다. NaN을 못 다루는 모델에서만 쓴다."""
    if 'prod_day' not in X:
        return X
    tr = X.loc[train_index, 'prod_day']
    X = X.copy()
    X['prod_day'] = X['prod_day'].fillna(tr[tr > 0].median())
    return X
