"""시간 최대 분석(02-4·03-1·06)의 정보 누설과 제출 파일을 자동으로 점검합니다.

실행: run_all.py로 노트북을 모두 실행한 뒤 `python -m pytest tests -q`
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
PRED = ROOT / 'outputs' / 'predictions' / 'test_predictions.csv'
PREDS = ROOT / 'outputs' / 'preds'


def need(path):
    if not path.exists():
        pytest.skip(f'{path.relative_to(ROOT)} 없음 — run_all.py로 노트북을 먼저 실행하세요')


@pytest.fixture(scope='module')
def mu():
    need(ROOT / 'data' / 'raw' / 'okm_augumented_2021.csv')
    need(ROOT / 'data' / 'processed' / 'days.csv')
    import hourly_utils
    return hourly_utils


def test_시간최대_피처는_예측시점_이후_값을_바꿔도_그대로다(mu):
    """미래 전력·가동 여부를 난수로 바꿔도 과거값 피처가 같아야 한다(트랙 A·B 각 10개 시점)."""
    m = mu.our_frame(mu.load_friend()['days'])
    assert mu.pu.check_no_leak(m, 'A', 'S1', n_checks=10) == 10
    assert mu.pu.check_no_leak(m, 'B', 'S1', n_checks=10) == 10


def test_제출파일의_기간과_열():
    need(PRED)
    f = pd.read_csv(PRED, encoding='utf-8-sig', parse_dates=['시각'])
    assert f['시각'].min() == pd.Timestamp('2021-08-09 00:00') and f['시각'].max() == pd.Timestamp('2021-09-14 23:00')
    assert f['시각'].is_unique and len(f) == 37 * 24
    for c in ['실제_시간최대_kW', '1시간앞_15분모델_kW', '1시간앞_시간최대_q50_kW', '1시간앞_피크확률_보정',
              '경보_기본', '경보_재현율우선', '하루앞_15분모델_kW', '하루앞_시간최대_q50_kW']:
        assert c in f.columns


def test_경보는_예측값만으로_다시_만들어진다():
    """경보 열이 실제값이 아니라 같은 행의 예측값과 준비선(178.7kW)만으로 재현되어야 한다."""
    need(PRED)
    f = pd.read_csv(PRED, encoding='utf-8-sig')
    ok = f['1시간앞_15분모델_kW'].notna()
    assert ((f.loc[ok, '1시간앞_15분모델_kW'] >= 178.7) == f.loc[ok, '경보_기본'].astype(bool)).all()
    assert ((f['1시간앞_시간최대_q50_kW'] >= 178.7) == f['경보_재현율우선'].astype(bool)).all()


def test_확률보정은_앞주로만_맞추므로_1주차에는_값이_없다():
    need(PRED)
    f = pd.read_csv(PRED, encoding='utf-8-sig')
    assert f.loc[f['주차'] == '1주차', '1시간앞_피크확률_보정'].isna().all()
    assert f.loc[f['주차'] != '1주차', '1시간앞_피크확률_보정'].notna().all()
    p = f['1시간앞_피크확률_보정'].dropna()
    assert ((p >= 0) & (p <= 1)).all()


def test_주별_학습_예측은_시험기간에만_있다():
    need(PREDS / 'oof_A.parquet')
    for tr in 'AB':
        o = pd.read_parquet(PREDS / f'oof_{tr}.parquet')
        assert o.index.min() >= pd.Timestamp('2021-08-09') and o.index.max() <= pd.Timestamp('2021-09-14 23:00')
        assert set(o['week']) == {'1주차', '2주차', '3주차', '4주차', '5주차'}
