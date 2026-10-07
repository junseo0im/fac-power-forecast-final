"""가이드북 베이스라인(RNN·RF) 재평가용 함수(원래 수치 읽기, 168시차, 열별 MinMax, RNN 구조, 재귀 예측, RF 피처)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .data import input_series
from .features import GROUP_CAL, GROUP_LAG, KIND_LABEL, _avail_reason, assert_feature_availability


BASELINE_NB = ROOT / 'notebooks' / 'resource_optimization_baseline.ipynb'


def guidebook_reference(path: Path | str = BASELINE_NB) -> pd.DataFrame:
    """베이스라인 노트북에 저장된 실행 결과('Baseline 결과 요약' 표)를 읽어 온다.

    가이드북 원래 수치를 손으로 옮겨 적지 않고 실행 출력에서 그대로 가져오기 위한 함수다.
    """
    import re
    nb = json.loads(Path(path).read_text(encoding='utf-8'))
    for cell in nb['cells']:
        for o in cell.get('outputs', []):
            html = ''.join(o.get('data', {}).get('text/html', ''))
            if 'SimpleRNN' in html and 'MAE' in html:
                rows = re.findall(r'<tr>\s*<th>\d+</th>(.*?)</tr>', html, flags=re.S)
                data = [[v.strip() for v in re.findall(r'<td>(.*?)</td>', r, flags=re.S)] for r in rows]
                ref = pd.DataFrame(data, columns=['모델', '평가 데이터', '샘플 수', 'MSE', 'RMSE', 'MAE'])
                return ref.astype({'샘플 수': int, 'MSE': float, 'RMSE': float, 'MAE': float})
    raise ValueError('베이스라인 노트북에서 결과 요약 표를 찾지 못했습니다')


class GuidebookScaler:
    """가이드북 [코드 46~47]처럼 '정답 1열 + 시차 168열'을 열마다 0~1로 바꾸는 MinMaxScaler.

    가이드북은 전체 구간(테스트 포함)으로 fit했지만, 여기서는 학습 구간 행으로만 fit해 테스트 정보가 새지 않게 한다.
    """

    def fit(self, y: np.ndarray, lags: np.ndarray):
        from sklearn.preprocessing import MinMaxScaler
        self.sc = MinMaxScaler(feature_range=(0, 1)).fit(np.column_stack([y, lags]))
        return self

    def transform_y(self, y):
        return np.asarray(y) * self.sc.scale_[0] + self.sc.min_[0]

    def inverse_y(self, y_scaled):
        return (np.asarray(y_scaled) - self.sc.min_[0]) / self.sc.scale_[0]

    def transform_lags(self, lags):
        return np.asarray(lags) * self.sc.scale_[1:] + self.sc.min_[1:]


def build_guidebook_rnn():
    """가이드북 [코드 53]과 같은 구조의 RNN을 만든다: 입력 (24, 7) → SimpleRNN 50 → Flatten → Dense 1, Adam(lr=1e-4), MSE.

    Keras 3의 torch 백엔드를 쓴다(구조는 같고 TensorFlow가 필요 없음). keras를 처음 불러오기 전에 백엔드를 정해야 한다.
    입력 (24, 7)은 시차 168개 [t-1, ..., t-168]을 앞에서부터 7개씩 자른 모양이다(가이드북 [코드 51]).
    """
    os.environ.setdefault('KERAS_BACKEND', 'torch')
    import keras
    model = keras.Sequential([keras.Input(shape=(24, 7)), keras.layers.SimpleRNN(50),
                              keras.layers.Flatten(), keras.layers.Dense(1)])
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=1e-4))
    return model


def recursive_forecast(predict_one, history: np.ndarray, steps: int = 24) -> np.ndarray:
    """1시간 앞 모델로 여러 시간을 재귀 예측한다(트랙 B용).

    history: origin마다 직전 168개 값(오래된 → 최근 순서)을 한 행에 담은 배열
    predict_one(lags): [t-1, ..., t-168] 순서의 시차 행렬을 받아 1시간 앞 예측값 벡터를 돌려주는 함수
    예측값을 다음 입력의 맨 끝에 붙여 steps번 반복하므로, 앞 시간의 오차가 뒤 시간 입력에 그대로 쌓인다.
    """
    win = np.asarray(history, float).copy()
    out = []
    for _ in range(steps):
        p = np.asarray(predict_one(win[:, ::-1]), float).ravel()
        out.append(p)
        win = np.column_stack([win[:, 1:], p])
    return np.column_stack(out)


RF_FEATURES = ['시간', '기온', '풍속', '습도', '강수량', 'day', 'd', 'm', 'y_lag1']


def guidebook_rf_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """가이드북 RF [코드 72] 수정판 피처를 시간 단위 트랙 A로 옮긴다.

    - 가이드북: 현재 15분 행의 (시간, 기온, 풍속, 습도, 강수량, day, d, m) + 현재 15분 값 → 다음 15분 값
    - 시간 단위: 직전 시각(t-1) 행의 같은 8개 + 직전 시각 시간 피크 y(t-1) → 대상 시각 y(t)
    - `시간`은 재구성한 시각을 쓴다(7/13·7/15 오류 보정). 기상 결측은 가이드북 [코드 12]처럼 0으로 채운다.
    모든 값이 1시간 전 정보라 트랙 A 규칙을 지킨다(가용성 표로 확인).
    """
    prev = df.shift(1)
    X = pd.DataFrame({'시간': prev['hour'], '기온': prev['기온'].fillna(0), '풍속': prev['풍속'].fillna(0),
                      '습도': prev['습도'].fillna(0), '강수량': prev['강수량'].fillna(0),
                      'day': prev['day'], 'd': prev['d'], 'm': prev['m'],
                      'y_lag1': input_series(df).shift(1)}, index=df.index).astype(float)
    desc = {'시간': '직전 시각(0~23)', '기온': '직전 시각 기온 실측', '풍속': '직전 시각 풍속 실측',
            '습도': '직전 시각 습도 실측', '강수량': '직전 시각 강수량 실측', 'day': '직전 시각 요일(1=월)',
            'd': '직전 시각 일', 'm': '직전 시각 월', 'y_lag1': '직전 시각 시간 피크'}
    cal = {'시간', 'day', 'd', 'm'}
    avail = pd.DataFrame([{'트랙': 'A', '시나리오': 'S0', '피처': c,
                           '그룹': GROUP_CAL if c in cal else GROUP_LAG,
                           'kind': 'known' if c in cal else 'lag_h', '최소 시차': None if c in cal else 1,
                           '설명': desc[c]} for c in RF_FEATURES])
    assert_feature_availability(avail, 'A', 'S0')
    avail['종류'] = avail['kind'].map(KIND_LABEL)
    avail['예측 시점에 알 수 있는 이유'] = avail.apply(lambda r: _avail_reason(r, 'A'), axis=1)
    return X, avail
