"""정보 누설(답을 미리 보는 일)과 평가 범위를 자동으로 점검합니다.

실행: 노트북을 모두 실행한 뒤 `python -m pytest tests -q`
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "okm_augumented_2021.csv"
SLOTS = ROOT / "data" / "processed" / "slots_15min.csv"
FINAL_DAY = ROOT / "outputs" / "predictions" / "final_day_ahead.csv"
FINAL_HOUR = ROOT / "outputs" / "predictions" / "final_hour_ahead.csv"
P = ["15분", "30분", "45분", "60분"]
TEST_STARTS = pd.to_datetime(["2021-08-09", "2021-08-16", "2021-08-23", "2021-08-30", "2021-09-06"])
TEST_ENDS = list(TEST_STARTS[1:]) + [pd.Timestamp("2021-09-15")]
EXCLUDED = pd.to_datetime(["2021-07-13", "2021-07-15", "2021-07-28", "2021-07-30"])


def need(path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} 없음 — 원본 데이터를 넣고 노트북을 먼저 실행하세요")


def load_slots():
    need(SLOTS)
    s = pd.read_csv(SLOTS, encoding="utf-8-sig", parse_dates=["ts", "date"], index_col="ts")
    s["칸번호"] = (s["시각"] * 4).astype(int)
    s["하루종류코드"] = s["하루종류"].map({"평일가동": 0, "토요일": 1, "휴일": 2})
    return s


def test_공장인원과_평균은_정답에서_계산된_값이다():
    """금지 이유 확인: 공장인원 = 생산량 ÷ 그 시간 전력 합, 평균 = 전력 4칸의 평균(반올림)"""
    need(RAW)
    raw = pd.read_csv(RAW, encoding="utf-8-sig")
    has_prod = raw["생산량"] > 0
    calc = raw.loc[has_prod, "생산량"] / raw.loc[has_prod, P].sum(axis=1)
    assert np.isclose(calc, raw.loc[has_prod, "공장인원"], rtol=1e-6).all()
    assert ((raw[P].mean(axis=1) - raw["평균"]).abs() <= 0.5).all()   # 평균 열은 반올림된 정수


def test_전처리_데이터에_금지_열이_없다():
    """모델은 전처리된 15분 데이터만 읽으므로, 여기에 금지 열이 없으면 입력으로 쓸 수 없다"""
    s = load_slots()
    assert "공장인원" not in s.columns and "평균" not in s.columns


def test_사용_기간과_제외일():
    s = load_slots()
    assert s.index.min() == pd.Timestamp("2021-07-01 00:00") and s.index.max() == pd.Timestamp("2021-09-14 23:45")
    assert not s.loc[s["date"].isin(EXCLUDED), "정답사용"].any(), "제외한 날이 평가 대상에 들어 있음"
    assert s.loc[s["date"].isin(EXCLUDED[:2]), "전력"].isna().all(), "시간 순서가 깨진 날의 전력이 남아 있음"


def test_하루앞_예측은_시험주_이전_데이터만으로_만든_기준곡선과_같다():
    s = load_slots()
    need(FINAL_DAY)
    final = pd.read_csv(FINAL_DAY, encoding="utf-8-sig", parse_dates=["ts"], index_col="ts")
    keys = ["하루종류코드", "전날가동", "칸번호"]
    for st, en in zip(TEST_STARTS, TEST_ENDS):
        past = s[(s.index < st) & s["정답사용"]]
        curve = past.groupby(keys)["전력"].mean().rename("c")
        week = s[(s.index >= st) & (s.index < en)]
        expected = week.join(curve, on=keys)["c"]
        got = final.loc[week.index, "하루앞_예측"]
        assert np.allclose(expected.values, got.values, equal_nan=True), f"{st:%m/%d} 주 예측이 과거 데이터만으로 재현되지 않음"


def test_1시간앞_예측은_시험기간에만_있고_발행시각_이후를_예측한다():
    need(FINAL_HOUR)
    h = pd.read_csv(FINAL_HOUR, encoding="utf-8-sig", parse_dates=["발행시각", "대상시각"])
    assert (h["발행시각"] >= TEST_STARTS[0]).all() and (h["대상시각"] < TEST_ENDS[-1]).all()
    lead = h["대상시각"] - h["발행시각"]
    assert (lead >= pd.Timedelta(0)).all() and (lead <= pd.Timedelta(minutes=45)).all()
    assert (h["발행시각"].dt.minute == 0).all(), "예측은 매 정시에만 발행"
    assert not h.loc[h["대상시각"].dt.normalize().isin(EXCLUDED), "정답사용"].any()


DECISION_LOG = ROOT / "outputs" / "predictions" / "decision_log.csv"


def test_도우미_판단은_발행시각_이전에_알_수_있는_정보만_쓴다():
    """판단 기록의 예측값·판단이 같은 발행시각의 1시간 앞 예측(과 그 이전에 정해진 하루 앞 예측)만으로 다시 만들어지는지"""
    need(DECISION_LOG); need(FINAL_HOUR); need(FINAL_DAY)
    log = pd.read_csv(DECISION_LOG, encoding="utf-8-sig", parse_dates=["발행시각", "예측최대_시각"])
    h = pd.read_csv(FINAL_HOUR, encoding="utf-8-sig", parse_dates=["발행시각", "대상시각"])
    d = pd.read_csv(FINAL_DAY, encoding="utf-8-sig", parse_dates=["ts"], index_col="ts")
    s = load_slots()
    ready = 0.85 * s.loc[(s.index < TEST_STARTS[0]) & s["정답사용"], "전력"].max() - 10
    assert log["발행시각"].min() >= TEST_STARTS[1], "여유값을 고른 1주차가 판단 기록에 들어 있음"
    g = h.groupby("발행시각")
    pmax = g["1시간앞_예측"].max().reindex(log["발행시각"])
    assert np.allclose(log["1시간앞_예측최대"].values, pmax.round(1).values), "판단에 쓴 예측값이 같은 발행시각의 예측과 다름"
    n_alarm = g["1시간앞_예측"].apply(lambda x: int((x >= ready).sum())).reindex(log["발행시각"])
    assert (log["경보칸_수"].values == n_alarm.values).all()
    assert ((log["판단"] == "준비").values == (n_alarm.values > 0)).all(), "판단이 예측값 규칙으로 다시 만들어지지 않음"
    lead = log["예측최대_시각"] - log["발행시각"]
    assert ((lead >= pd.Timedelta(0)) & (lead <= pd.Timedelta(minutes=45))).all()
    dmax = [d.loc[t:t + pd.Timedelta(minutes=45), "하루앞_예측"].max() for t in log["발행시각"]]
    assert np.allclose(log["하루앞_예측최대"].values, np.round(dmax, 1)), "하루 앞 예측값이 저장된 예측과 다름"
    prev = s.groupby("date")["전날가동"].first().reindex(log["발행시각"].dt.normalize())
    assert (log["주의일"].values == ~prev.astype(bool).values).all(), "주의일이 전날 가동 여부(미리 아는 값)와 다름"
