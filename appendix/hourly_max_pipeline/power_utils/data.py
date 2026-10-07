"""데이터 불러오기와 플래그: 시각 재구성, 시간 피크 y, 0값(outage)·가동일·공휴일·복사일·생산계획, 한전 TOU, 15분 long 시계열."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PATH, HOLIDAYS_2021, OP_THRESHOLD, QUARTER_COLS, RNN_LAGS, TEST_START, WEATHER_COLS


def tou_2021(ts, holiday, saturday_rule: bool = True, bands: str = 'pre2023') -> pd.DataFrame:
    """한전 계절·시간대(TOU) 구분을 돌려준다. 기본은 2021년 데이터에 맞는 2023-01-01 개정 전 시간대다.

    - 계절: 여름 6~8월, 봄가을 3~5·9~10월, 겨울 11~2월
    - bands='pre2023'(기본): 경부하 23~09시. 여름·봄가을 최대부하 10~12·13~17시, 겨울 최대부하 10~12·17~20·22~23시
    - bands='2023': 2023-01-01부터 적용된 현행 시간대(민감도 분석용). 경부하 22~08시,
      여름·봄가을 최대부하 11~12·13~18시, 겨울 최대부하 09~12·16~19시
    - 나머지 시간은 중간부하. 공휴일(holiday=True)과 일요일은 하루 종일 경부하로 계량한다
      (약관의 공휴일 = 「관공서의 공휴일에 관한 규정」, 일요일 포함, 임시공휴일 제외).
    - saturday_rule=True면 공휴일이 아닌 토요일의 최대부하 시간을 중간부하로 바꾼다
      (약관상 사용전력량 기준. 중간·최대부하를 함께 보는 과금 시간에는 영향이 없다).
    """
    ts = pd.DatetimeIndex(ts)
    h, m, dow = ts.hour.to_numpy(), ts.month.to_numpy(), ts.dayofweek.to_numpy()
    holiday = np.asarray(holiday, dtype=bool)
    season = np.select([np.isin(m, [6, 7, 8]), np.isin(m, [3, 4, 5, 9, 10])], ['여름', '봄가을'], '겨울')
    winter = season == '겨울'
    if bands == 'pre2023':
        peak = np.where(winter, np.isin(h, [10, 11, 17, 18, 19, 22]), np.isin(h, [10, 11, 13, 14, 15, 16]))
        off = (h >= 23) | (h < 9)
    elif bands == '2023':
        peak = np.where(winter, np.isin(h, [9, 10, 11, 16, 17, 18]), np.isin(h, [11, 13, 14, 15, 16, 17]))
        off = (h >= 22) | (h < 8)
    else:
        raise ValueError(bands)
    load = np.where(off, '경부하', np.where(peak, '최대부하', '중간부하')).astype(object)
    load[holiday | (dow == 6)] = '경부하'
    if saturday_rule:
        load[(dow == 5) & (load == '최대부하')] = '중간부하'
    out = pd.DataFrame({'season': season, 'load': load}, index=ts)
    out['season_code'] = out['season'].map({'겨울': 0, '봄가을': 1, '여름': 2})
    out['load_code'] = out['load'].map({'경부하': 0, '중간부하': 1, '최대부하': 2})
    return out


def load_data(path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """원본 csv를 읽어 시간 단위 표(df)와 15분 long 시계열(q15)을 만든다.

    - 시각은 '날짜 + 그날의 행 순서'로 재구성한다. 원본 `시간`이 깨진 48행은 time_err=True로 표시한다.
    - y = max(15분, 30분, 45분, 60분): 시간별 최대수요전력(주 타깃), usage = `평균` 컬럼(사용량)
    - outage = 15분 값 중 하나라도 0인 시간, outage_full = 4개 모두 0인 시간. y_obs는 outage를 NaN으로 바꾼 y
    - op_day = 09~17시 평균 > 80kW(주간 가동 여부, 실제 전력에서 역산한 S2 전용 오라클)
    - prod_day = 당일 생산량 합계. `시간`이 깨진 날(7/13·7/15)은 생산량도 0으로 결손돼 NaN, prod_missing=True
    - copied_day = 가동일인데 15분 값이 전날·1주 전과 20시간 이상 똑같은 날(4~5월 11일, 증강 데이터의 복사 흔적)
    - 15분 long 시계열의 ts는 각 구간의 시작 시각이다('15분' 값 = 정시~15분 구간).
    """
    raw = pd.read_csv(path, encoding='utf-8-sig')
    date = pd.to_datetime(raw['날짜'].astype(str), format='%Y%m%d')
    hour = date.groupby(date).cumcount()
    df = raw.copy()
    df.index = pd.DatetimeIndex(date + pd.to_timedelta(hour, unit='h'), name='ts')
    df[QUARTER_COLS + ['평균'] + WEATHER_COLS] = df[QUARTER_COLS + ['평균'] + WEATHER_COLS].astype(float)
    df['date'] = date.to_numpy()
    df['hour'] = hour.to_numpy()
    assert df.index.is_unique and (df.groupby('date').size() == 24).all(), '날짜마다 24행이어야 합니다'
    df['time_err'] = df['시간'].to_numpy() != df['hour'].to_numpy()

    q = df[QUARTER_COLS]
    df['y'] = q.max(axis=1)
    df['usage'] = df['평균']
    df['outage_full'] = (q == 0).all(axis=1)
    df['outage'] = (q == 0).any(axis=1)
    df['y_obs'] = df['y'].where(~df['outage'])

    df['dow'] = df.index.dayofweek          # 0=월 ~ 6=일
    df['month'] = df.index.month
    df['weekend'] = df['dow'] >= 5
    hol_dates = df['date'].dt.strftime('%Y-%m-%d')
    df['holiday'] = hol_dates.isin(list(HOLIDAYS_2021))
    df['holiday_name'] = hol_dates.map(HOLIDAYS_2021)
    df['shift'] = df['hour'].between(9, 17)          # 주간 교대(인건비 1.0인 시간)
    df['인건비_원본'] = df['인건비']
    df['인건비'] = np.where(df['shift'], 1.0, 1.5)   # 재구성한 시각으로 다시 계산

    tou = tou_2021(df.index, df['holiday'])
    df['tou_season'], df['tou_load'] = tou['season'].to_numpy(), tou['load'].to_numpy()
    df['tou_season_code'], df['tou_load_code'] = tou['season_code'].to_numpy(), tou['load_code'].to_numpy()
    df['billing_hour'] = df['tou_load'].isin(['중간부하', '최대부하'])
    # 민감도 분석용: 2023-01-01 개정 시간대 기준 과금 시간(08시가 중간부하로 바뀌어 기동 피크가 포함됨)
    df['billing_hour_2023'] = tou_2021(df.index, df['holiday'], bands='2023')['load'].isin(['중간부하', '최대부하']).to_numpy()

    # 일 단위 플래그 → 시간 단위로 펼침
    day = pd.DataFrame({'m917': df[df['shift']].groupby('date')['usage'].mean()})
    day['op_day'] = day['m917'] > OP_THRESHOLD
    day['after_shutdown'] = day['op_day'] & ~day['op_day'].shift(1, fill_value=True).astype(bool)
    day['prod_missing'] = df.groupby('date')['time_err'].any()
    day['prod_day'] = df.groupby('date')['생산량'].sum().astype(float).where(~day['prod_missing'])
    # 복사일: 가동일인데 15분 값 4개가 전날(24h) 또는 1주 전(168h)과 20시간 이상 완전히 같은 날(증강 데이터 흔적)
    same = {k: pd.Series((q.values == q.shift(k).values).all(axis=1), index=df.index) & ~df['outage'] for k in (24, 168)}
    n_same = pd.concat([s.groupby(df['date']).sum() for s in same.values()], axis=1).max(axis=1)
    day['copied_day'] = day['op_day'] & (n_same >= 20)
    for col in ('op_day', 'after_shutdown', 'prod_missing', 'prod_day', 'copied_day'):
        df[col] = df['date'].map(day[col]).to_numpy()
    df['prod_bin'] = _prod_bin(df, day)

    q15 = _to_quarter_long(df)
    return df, q15


def _prod_bin(df: pd.DataFrame, day: pd.DataFrame) -> np.ndarray:
    """생산계획(일 생산량) 구간: 없음/낮음/중간/높음/결손. 구간 경계는 테스트 이전 생산일로만 정한다."""
    pre = day.loc[(day.index < TEST_START) & (day['prod_day'] > 0), 'prod_day']
    lo, hi = pre.quantile([1 / 3, 2 / 3])
    p = df['prod_day']
    return np.select([p.isna(), p == 0, p <= lo, p <= hi], ['결손', '없음', '낮음', '중간'], '높음')


def _to_quarter_long(df: pd.DataFrame) -> pd.DataFrame:
    """1시간 1행(15·30·45·60분)을 15분 1행으로 펼친다(6,168 × 4 = 24,672행)."""
    parts = []
    for k, col in enumerate(QUARTER_COLS):
        parts.append(pd.DataFrame({
            'ts': df.index + pd.Timedelta(minutes=15 * k),
            'hour_ts': df.index, 'quarter': k + 1, 'value': df[col].to_numpy(),
            'outage_hour': df['outage'].to_numpy()}))
    q15 = pd.concat(parts).sort_values('ts').reset_index(drop=True)
    q15['zero'] = q15['value'] == 0
    return q15


def input_series(df: pd.DataFrame) -> pd.Series:
    """모델 입력용 시간 피크: outage(NaN)는 직전 값으로 채운다. 과거 방향으로만 채우므로 누수가 없다."""
    return df['y_obs'].ffill()


def lag_matrix(s: pd.Series, n_lags: int = RNN_LAGS) -> np.ndarray:
    """각 시각 t의 [s(t-1), s(t-2), ..., s(t-n)] 행렬을 만든다(가이드북 't-1'~'t-168' 컬럼과 같은 순서)."""
    v = s.to_numpy(float)
    out = np.full((len(v), n_lags), np.nan)
    for j in range(1, n_lags + 1):
        out[j:, j - 1] = v[:-j]
    return out


_DF_CACHE = {}


def load_data_cached() -> pd.DataFrame:
    """load_data()의 시간 단위 표를 한 번만 읽어 재사용한다."""
    if 'df' not in _DF_CACHE:
        _DF_CACHE['df'] = load_data()[0]
    return _DF_CACHE['df']
