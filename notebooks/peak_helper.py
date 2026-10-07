"""계산 도구를 골라 쓰는 규칙 기반 질의응답 도우미 (피크 관리 도우미의 확장, 노트북 04-2의 6절).

질문 → 거절 확인 → 의도 분류(키워드 점수 + 필수 근거 단어) → 슬롯 추출(정규식)
     → 도구 계획(고정표) → 도구 실행 → 문장 틀 답변 + 근거 카드 + 감사 기록

- 외부 서비스·새 패키지를 쓰지 않습니다. 같은 질문과 같은 '지금 시각'이면 언제나 같은 답이 나옵니다.
- 답변의 숫자는 모두 도구 출력에서만 나옵니다. 도구는 저장된 결과(outputs/…)를 읽거나 노트북 04-1·04-2의 로직을 그대로 옮긴 계산만 합니다.
- '지금 시각'(as_of) 이후의 실제 전력은 돌려주지 않습니다(노트북 04-2의 '판단 시각까지의 정보만' 원칙).
- 이 도우미는 제안만 합니다. 설비를 끄거나 설정을 바꾸는 도구는 없습니다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "slots": ROOT / "data" / "processed" / "slots_15min.csv",
    "hour": ROOT / "outputs" / "predictions" / "final_hour_ahead.csv",
    "day": ROOT / "outputs" / "predictions" / "final_day_ahead.csv",
    "dlog": ROOT / "outputs" / "predictions" / "decision_log.csv",
    "t42": ROOT / "outputs" / "tables" / "t42_몬테카를로_강건성.csv",
}

FIRST_TEST, TEST_END = pd.Timestamp("2021-08-09"), pd.Timestamp("2021-09-15")
SUMMER_START = pd.Timestamp("2021-07-12")
MARGIN, BASE = 10, 22                       # 02-3 노트북의 여유값, 04-1 노트북의 기본 전력
WEEK_STARTS = pd.to_datetime(["2021-08-16", "2021-08-23", "2021-08-30", "2021-09-06"])
TARIFF_BASIC = 7220                         # 원/kW·월, 산업용(을) 고압A 선택Ⅰ (노트북 04-1, 2021년 요금표)
TARIFF_KWH = {"여름": (56.6, 109.5, 191.6), "봄가을": (56.6, 79.1, 109.8)}
BAND_NAME = ["경부하", "중간부하", "최대부하"]
HOLIDAYS_2021 = set(pd.to_datetime(["2021-01-01", "2021-02-11", "2021-02-12", "2021-02-13", "2021-03-01", "2021-05-05",
                                    "2021-05-19", "2021-06-06", "2021-08-15", "2021-08-16"]))   # 요금 계산용 공휴일(노트북 04-1과 같음)
W_PEAK = 1e5


class ToolError(Exception):
    """도구가 답할 수 없는 입력 (기간 밖, 아직 모르는 정보, 입력 부족 등)"""


# ================================================================ 데이터 (처음 쓸 때 한 번 읽음)
class _Data:
    def __init__(self):
        missing = [k for k, p in FILES.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"필요한 파일이 없습니다: {missing} — 노트북 01-1~04-2를 먼저 실행하세요")
        self.slots = pd.read_csv(FILES["slots"], encoding="utf-8-sig", parse_dates=["ts", "date"], index_col="ts")
        self.hour = pd.read_csv(FILES["hour"], encoding="utf-8-sig", parse_dates=["발행시각", "대상시각"])
        self.day = pd.read_csv(FILES["day"], encoding="utf-8-sig", parse_dates=["ts"], index_col="ts")["하루앞_예측"]
        self.dlog = pd.read_csv(FILES["dlog"], encoding="utf-8-sig", parse_dates=["발행시각", "예측최대_시각"])
        self.t42 = pd.read_csv(FILES["t42"], encoding="utf-8-sig")
        s = self.slots
        self.TARGET = float(s[(s.index < FIRST_TEST) & s["정답사용"]]["전력"].max())   # 04·05·04-2 노트북과 같음
        self.L1, self.L2 = 0.85 * self.TARGET, 0.90 * self.TARGET
        self.READY = self.L1 - MARGIN                                                   # 준비선 = 받는 칸 상한
        self.daily = s.groupby("date")[["하루종류", "전날가동"]].first()
        h = self.hour
        self.past_prec = {}                     # 노트북 04-2: 그 주 이전 기록으로 본 경보 적중 비율
        for ws in WEEK_STARTS:
            p = h[(h["대상시각"] >= FIRST_TEST) & (h["대상시각"] < ws) & h["정답사용"]]
            a = p["1시간앞_예측"] >= self.READY
            self.past_prec[ws] = (a & (p["전력"] >= self.L1)).sum() / a.sum()


@lru_cache(maxsize=None)
def data() -> _Data:
    return _Data()


def files_ready() -> bool:
    return all(p.exists() for p in FILES.values())


def r1(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 1)


def basis():
    D = data()
    return {"목표전력": r1(D.TARGET), "1단계": r1(D.L1), "2단계": r1(D.L2), "준비선": r1(D.READY), "여유값": MARGIN}


def _check_date(d):
    if d is None:
        raise ToolError("날짜가 필요합니다")
    d = pd.Timestamp(d).normalize()
    if not (FIRST_TEST <= d < TEST_END):
        raise ToolError(f"{d:%m/%d}는 예측이 저장된 시험 기간(08/09~09/14) 밖입니다")
    return d


def _visible(ts, as_of):
    return as_of is None or pd.Timestamp(ts) <= pd.Timestamp(as_of)


def is_risk(t):
    return t.weekday() < 5 and (8 <= t.hour < 12 or 13 <= t.hour < 16)


def _receivers_mask(idx, prev_ran):
    """노트북 04-1 receivers(): 06~22시, 위험 시간대·16시대·점심(12시대)·저녁 휴게(17:00~17:15) 제외, 전날 쉰 날은 06~08시 제외"""
    h, mi = idx.hour, idx.minute
    risk = ((h >= 8) & (h < 12)) | ((h >= 13) & (h < 16))
    m = (h >= 6) & (h < 22) & ~risk & (h != 16) & ~((h == 12) | ((h == 17) & (mi < 30)))
    if not prev_ran:
        m &= h >= 8
    return np.asarray(m)


def _unit_price(ts):
    """노트북 04-1 tou_band(): 일요일·공휴일은 하루 종일 경부하, 토요일의 최대부하 시간은 중간부하"""
    h = ts.hour
    if ts.weekday() == 6 or ts.normalize() in HOLIDAYS_2021 or h >= 23 or h < 9:
        band = 0
    elif (10 <= h < 12) or (13 <= h < 17):
        band = 1 if ts.weekday() == 5 else 2
    else:
        band = 1
    return TARIFF_KWH["여름" if ts.month in (6, 7, 8) else "봄가을"][band], band


# ================================================================ 도구 1. 예측 조회
def forecast(date, kind="day", hour_=None, as_of=None):
    """kind='day': 전날 만든 하루 앞 예측 / kind='hour': 그 정시에 발행된 1시간 앞 예측 4칸"""
    D = data(); d = _check_date(date)
    if kind == "hour":
        if hour_ is None:
            raise ToolError("1시간 앞 예측은 몇 시 발행인지 필요합니다")
        t = d + pd.Timedelta(hours=int(hour_))
        if not _visible(t, as_of):
            raise ToolError(f"{t:%m/%d %H:%M} 발행 예측은 지금({pd.Timestamp(as_of):%m/%d %H:%M}) 아직 나오지 않았습니다")
        g = D.hour[D.hour["발행시각"] == t].sort_values("대상시각")
        if g.empty:
            raise ToolError("그 시각의 1시간 앞 예측이 없습니다")
        i = g["1시간앞_예측"].idxmax()
        return {"종류": "1시간 앞", "발행시각": f"{t:%m/%d %H:%M}",
                "칸별예측": {f"{a:%H:%M}": r1(v) for a, v in zip(g["대상시각"], g["1시간앞_예측"])},
                "예측최대": r1(g.loc[i, "1시간앞_예측"]), "예측최대_시각": f"{g.loc[i, '대상시각']:%H:%M}",
                "준비선까지": r1(D.READY - g.loc[i, "1시간앞_예측"]), "기준": basis()}
    g = D.day[D.day.index.normalize() == d]
    if hour_ is not None:
        g = g[g.index.hour == int(hour_)]
    i = g.idxmax(); hmax = g.groupby(g.index.hour).max()
    risky = [int(h) for h, v in hmax.items() if v >= D.READY]
    return {"종류": "하루 앞", "날짜": f"{d:%m/%d}", "범위": "하루 전체" if hour_ is None else f"{int(hour_):02d}시대",
            "예측최대": r1(g.max()), "예측최대_시각": f"{i:%H:%M}", "준비선이상_시간": risky,
            "주의일": not bool(D.daily.loc[d, "전날가동"]), "하루종류": D.daily.loc[d, "하루종류"], "기준": basis()}


# ================================================================ 도구 2. 경보 판단 (노트북 04-2 규칙)
def alarm(date, hour_, as_of=None):
    D = data(); d = _check_date(date)
    if hour_ is None:
        raise ToolError("몇 시 판단인지 필요합니다")
    t = d + pd.Timedelta(hours=int(hour_))
    if not _visible(t, as_of):
        raise ToolError(f"{t:%m/%d %H:%M} 판단은 아직 내릴 수 없습니다(지금 {pd.Timestamp(as_of):%H:%M})")
    g = D.hour[D.hour["발행시각"] == t].sort_values("대상시각")
    if g.empty:
        raise ToolError("그 시각의 1시간 앞 예측이 없습니다")
    ws = [w for w in WEEK_STARTS if w <= t]
    if not ws:
        raise ToolError("1주차(08/09~08/15)는 여유값을 고른 주라 판단 기록이 없습니다")
    prec = float(D.past_prec[max(ws)])
    pred = g["1시간앞_예측"]; al = pred >= D.READY; i = pred.idxmax()
    out = {"발행시각": f"{t:%m/%d %H:%M}", "판단": "준비" if al.any() else "정상",
           "경보칸_수": int(al.sum()), "경보칸_시각": [f"{x:%H:%M}" for x in g.loc[al, "대상시각"]],
           "예측최대": r1(pred.max()), "예측최대_시각": f"{g.loc[i, '대상시각']:%H:%M}",
           "준비선까지": r1(D.READY - pred.max()), "1단계까지": r1(D.L1 - pred.max()),
           "주의일": not bool(D.daily.loc[d, "전날가동"]), "위험시간대": is_risk(t),
           "적중비율_이전기록": round(prec, 3), "N번중1번": max(1, round(1 / prec)), "기준": basis()}
    row = D.dlog[D.dlog["발행시각"] == t]       # 노트북 04-2 판단 기록과 대조
    out["판단기록과_일치"] = (bool(row["판단"].iloc[0] == out["판단"] and abs(row["1시간앞_예측최대"].iloc[0] - out["예측최대"]) < 0.051)
                          if len(row) else None)
    return out


# ================================================================ 도구 3a. What-if: 덩어리 부하를 X시에 넣으면
def whatif_block(date, start, hours=2.0, kw=None, as_of=None):
    """하루 앞 예측(계획 기준)에 kw짜리 덩어리를 start부터 hours 동안 더해 봄.
    사후 확인(실제 전력 + 덩어리, 여름 최대 변화)은 지금 시각이 그날 이후일 때만 돌려줌."""
    D = data(); d = _check_date(date)
    if kw is None:
        raise ToolError("덩어리 부하의 크기(kW)가 필요합니다. 설비별 기록이 없어 도우미가 정할 수 없습니다(04-1 노트북의 '옮길 수 있는 부하' 점검 참고)")
    if start is None:
        raise ToolError("시작 시각이 필요합니다")
    st = d + pd.Timedelta(hours=float(start)); n = int(round(hours * 4))
    idx = pd.date_range(st, periods=n, freq="15min")
    if n < 1 or idx[-1] >= d + pd.Timedelta(days=1):
        raise ToolError("작업 구간이 그날 안에 들어가지 않습니다")
    full = D.day[D.day.index.normalize() == d]
    plan = full.copy(); plan.loc[idx] += kw
    win = plan.loc[idx]; rec = _receivers_mask(idx, bool(D.daily.loc[d, "전날가동"]))
    out = {"날짜": f"{d:%m/%d}", "시작": f"{st:%H:%M}", "길이_시간": hours, "크기_kW": kw,
           "넣은칸_예측최대": r1(win.max()), "넣은칸_예측최대_시각": f"{win.idxmax():%H:%M}",
           "준비선_넘는칸": int((win >= D.READY).sum()), "1단계_넘는칸": int((win >= D.L1).sum()),
           "준비선까지_여유": r1(D.READY - win.max()),
           "하루_예측최대_전": r1(full.max()), "하루_예측최대_후": r1(plan.max()),
           "받는칸_규칙위반칸": int((~rec).sum()), "전체칸": n,
           "위험시간대_걸림": bool(any(is_risk(t) for t in idx)), "기준": basis()}
    out["판정"] = "넣을 수 있음" if out["준비선_넘는칸"] == 0 and out["받는칸_규칙위반칸"] == 0 else "권하지 않음"
    if _visible(d + pd.Timedelta(days=1), as_of):
        s = D.slots
        act = s.loc[s["date"] == d, "전력"].copy(); act.loc[idx] += kw
        others = s.loc[(s.index >= SUMMER_START) & (s["date"] != d), "전력"]
        out.update({"사후_넣은칸_실제최대": r1(act.loc[idx].max()), "사후_1단계_넘는칸": int((act.loc[idx] >= D.L1).sum()),
                    "사후_하루최대": r1(act.max()), "사후_여름최대": r1(max(others.max(), act.max())),
                    "여름최대_원래": r1(s.loc[s.index >= SUMMER_START, "전력"].max())})
    return out


# ================================================================ 도구 3b. 덩어리 부하의 가장 좋은 시작 시각 (받는 칸 안 전수 탐색)
def best_block(date, hours=2.0, kw=None, as_of=None):
    D = data(); d = _check_date(date)
    if kw is None:
        raise ToolError("덩어리 부하의 크기(kW)가 필요합니다")
    n = int(round(hours * 4)); full = D.day[D.day.index.normalize() == d]
    prev = bool(D.daily.loc[d, "전날가동"]); cands = []
    for k in range(0, len(full) - n + 1):
        idx = full.index[k:k + n]
        if _receivers_mask(idx, prev).all():
            cands.append((r1((full.loc[idx] + kw).max()), idx[0]))
    if not cands:
        raise ToolError("받는 칸 안에 그 길이의 연속 구간이 없습니다")
    cands.sort(key=lambda x: (x[0], x[1]))
    return {"날짜": f"{d:%m/%d}", "길이_시간": hours, "크기_kW": kw, "후보수": len(cands),
            "최선_시작": f"{cands[0][1]:%H:%M}", "최선_예측최대": cands[0][0], "최선_준비선까지": r1(D.READY - cands[0][0]),
            "차선": [{"시작": f"{t:%H:%M}", "예측최대": v} for v, t in cands[1:3]],
            "최악_시작": f"{cands[-1][1]:%H:%M}", "최악_예측최대": cands[-1][0], "기준": basis()}


# ================================================================ 도구 3c. 받는 칸 LP (노트북 04-1 lp_receive와 같음)
def lp_receive(level, E, cost):
    """min W·Z + Σ cost·a   s.t. Σa = E,  level_t + a_t ≤ Z,  a ≥ 0"""
    n = len(level)
    A = lil_matrix((n + 1, n + 1)); lo = np.full(n + 1, -np.inf); hi = np.zeros(n + 1)
    for i in range(n):
        A[i, i] = 1; A[i, n] = -1; hi[i] = -level[i]
    A[n, :n] = 1; lo[n] = hi[n] = E
    res = milp(np.r_[cost, W_PEAK], constraints=LinearConstraint(A.tocsr(), lo, hi),
               bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.full(n + 1, np.inf)))
    assert res.success, res.message
    return res.x[:n]


def lp_plan(date, f=0.10, amount_from="forecast", as_of=None, observed_only=False, return_series=False):
    """위험 시간대 부하의 f를 받는 칸(06~22시)으로 옮기는 LP. 빼는 칸은 고정, 받는 칸 배분만 최적화.
    amount_from='forecast': 옮길 양을 하루 앞 예측으로 정함(계획용, 실제값을 쓰지 않음).
    amount_from='actual' + observed_only=True: 노트북 04-1과 똑같은 계산(받는 칸 재계획 결과 재현 검증용)."""
    D = data(); d = _check_date(date)
    if D.daily.loc[d, "하루종류"] != "평일가동":
        raise ToolError(f"{d:%m/%d}는 평일 가동일이 아니라 옮길 계획이 없습니다")
    g = D.slots[D.slots["date"] == d]; lv = D.day.loc[g.index]
    h = g.index.hour
    okm = g["전력"].notna().values if observed_only else np.ones(len(g), bool)
    donors = g.index[np.asarray(((h >= 8) & (h < 12)) | ((h >= 13) & (h < 16))) & okm]
    src = g.loc[donors, "전력"] if amount_from == "actual" else lv.loc[donors]
    amt = (src - BASE).clip(lower=0) * f
    R = g.index[_receivers_mask(g.index, bool(D.daily.loc[d, "전날가동"])) & okm]
    price_r = np.array([_unit_price(t)[0] for t in R]); price_d = np.array([_unit_price(t)[0] for t in donors])
    a = lp_receive(lv.loc[R].values.astype(float), float(amt.sum()), 0.25 * price_r)
    plan = lv.copy(); plan.loc[donors] -= amt.values; plan.loc[R] += a
    by_hour = pd.Series(a, index=R).groupby(R.hour).sum() * 0.25
    out = {"날짜": f"{d:%m/%d}", "옮긴비율_f": f, "옮길양_kWh": r1(amt.sum() * 0.25),
           "받는칸_시간별_kWh": {f"{int(k):02d}시": r1(v) for k, v in by_hour.items() if v > 0.05},
           "계획_받는칸_최대": r1((lv.loc[R] + a).max()), "계획_하루최대_전": r1(lv.max()), "계획_하루최대_후": r1(plan.max()),
           "전력량요금_변화_원": round(float((a * 0.25 * price_r).sum() - (amt.values * 0.25 * price_d).sum())), "기준": basis()}
    if _visible(d + pd.Timedelta(days=1), as_of):
        act = g["전력"].copy(); act.loc[donors] -= amt.values; act.loc[R] += a
        out["사후_하루최대_전"] = r1(g["전력"].max()); out["사후_하루최대_후"] = r1(act.max())
        if return_series:
            out["_series"] = act
    return out


# ================================================================ 도구 4. 몬테카를로 위험 조회 (몬테카를로 결과 t42를 읽기만 함)
def mc_risk(compliance=None, policy="균일 준수"):
    if compliance is None:
        raise ToolError("준수율(%)이 필요합니다. 몬테카를로 결과에는 80·90·95·100%가 있습니다")
    t = data().t42
    t = t[t["옮길 수 있는 부하"].str.startswith("f 5~15%") & (t["정책"] == policy)]
    if int(compliance) not in set(t["준수율(%)"]):
        raise ToolError(f"몬테카를로 결과에는 준수율 {sorted(set(t['준수율(%)']))}%만 있습니다(새로 계산하지 않음)")
    r = t[t["준수율(%)"] == int(compliance)].iloc[0]
    return {"정책": policy, "준수율": int(compliance), "기대절감_만원": float(r["기대 절감(만 원/년)"]),
            "하위5%_절감_만원": float(r["5% 분위 절감(만 원/년)"]), "절감0원_확률": float(r["절감 0원 확률(%)"]),
            "여름최대_중앙값": float(r["여름 최대 중앙값(kW)"]), "여름최대_95분위": float(r["여름 최대 95% 분위(kW)"]),
            "출처": "몬테카를로 결과 t42_몬테카를로_강건성.csv (4,000회, f 5~15%)"}


# ================================================================ 도구 5. 요금 계산 (2021년 요금표, 선택Ⅰ)
def tariff(kw=None, kwh=None, at=None):
    out = {"기본요금단가_원_kW월": TARIFF_BASIC, "계약": "산업용(을) 고압A 선택Ⅰ, 2021년 요금표"}
    if kw is not None:
        out.update({"피크_kW": kw, "연_기본요금_원": round(kw * TARIFF_BASIC * 12)})
    if kwh is not None and at is not None:
        price, band = _unit_price(pd.Timestamp(at))
        out.update({"전력량_kWh": kwh, "시간대": BAND_NAME[band], "단가_원_kWh": price, "전력량요금_원": round(kwh * price)})
    if len(out) == 2:
        raise ToolError("요금 계산에는 피크 kW(기본요금) 또는 kWh와 시각(전력량요금)이 필요합니다")
    return out


# ================================================================ 도구 6. 판단 기록 조회
def log_query(date, hour_=None, as_of=None):
    D = data(); d = _check_date(date)
    L = D.dlog[D.dlog["발행시각"].dt.normalize() == d]
    if L.empty:
        raise ToolError(f"{d:%m/%d}의 판단 기록이 없습니다(기록은 2~5주차 평일 가동일만)")
    if as_of is not None:
        L = L[L["발행시각"] <= pd.Timestamp(as_of)]
    if hour_ is not None:
        L = L[L["발행시각"].dt.hour == int(hour_)]
        if L.empty:
            raise ToolError("그 시각의 판단 기록이 아직 없습니다")
        r = L.iloc[0]
        out = {"발행시각": f"{r['발행시각']:%m/%d %H:%M}", "판단": r["판단"], "예측최대": r1(r["1시간앞_예측최대"]),
               "경보칸_수": int(r["경보칸_수"]), "조치문장": r["조치문장"], "근거문장": r["근거문장"]}
        if _visible(r["발행시각"] + pd.Timedelta(hours=1), as_of):
            out.update({"사후_실제최대": r1(r["실제_최대"]), "사후_1단계초과칸": int(r["실제_1단계초과칸"]),
                        "사후_경보로잡은칸": int(r["실제_경보로잡은칸"])})
        else:
            out["사후"] = "아직 지나지 않은 시간이라 실제값을 보여 주지 않습니다"
        return out
    out = {"날짜": f"{d:%m/%d}", "판단수": len(L), "준비수": int((L["판단"] == "준비").sum()),
           "준비_시각": [f"{t:%H}시" for t in L.loc[L["판단"] == "준비", "발행시각"]]}
    vis = L[[_visible(t + pd.Timedelta(hours=1), as_of) for t in L["발행시각"]]]
    if len(vis):
        miss = vis[vis["실제_1단계초과칸"] > vis["실제_경보로잡은칸"]]
        out.update({"사후_대상_판단수": len(vis), "사후_1단계초과칸": int(vis["실제_1단계초과칸"].sum()),
                    "사후_경보로잡은칸": int(vis["실제_경보로잡은칸"].sum()),
                    "사후_놓친칸": int((vis["실제_1단계초과칸"] - vis["실제_경보로잡은칸"]).sum()),
                    "사후_놓친칸이있던_시각": [f"{t:%H}시" for t in miss["발행시각"]],
                    "사후_헛경보_시각": [f"{t:%H}시" for t in vis.loc[(vis["판단"] == "준비") & (vis["실제_1단계초과칸"] == 0), "발행시각"]]})
    return out


TOOLS = {"forecast": forecast, "alarm": alarm, "whatif_block": whatif_block, "best_block": best_block,
         "lp_plan": lp_plan, "mc_risk": mc_risk, "tariff": tariff, "log_query": log_query}

TOOL_SPEC = pd.DataFrame([
    ["예측 조회", "forecast", "날짜, 종류(하루 앞/1시간 앞), 시각", "예측 최대·시각, 칸별 예측, 준비선 이상 시간", "final_day_ahead.csv, final_hour_ahead.csv"],
    ["경보 판단", "alarm", "날짜, 정시", "준비/정상, 경보 칸, 적중 비율(그 주 이전 기록)", "노트북 04-2 판단 규칙"],
    ["What-if (덩어리 부하)", "whatif_block", "날짜, 시작 시각, 길이, kW", "넣은 구간 예측 최대, 넘는 칸, 판정, (사후) 여름 최대", "노트북 04-1 받는 칸 규칙"],
    ["가장 좋은 시작 시각", "best_block", "날짜, 길이, kW", "받는 칸 안 최선·차선·최악 시각", "받는 칸 안 전수 탐색"],
    ["받는 칸 LP", "lp_plan", "날짜, 옮길 비율 f", "시간별 받는 양, 계획상 하루 최대 전·후", "노트북 04-1 lp_receive"],
    ["몬테카를로 위험", "mc_risk", "준수율, 정책", "기대 절감, 하위 5%, 절감 0원 확률", "t42_몬테카를로_강건성.csv (읽기만)"],
    ["요금 계산", "tariff", "kW 또는 kWh·시각", "연 기본요금, 전력량요금", "노트북 04-1 요금표"],
    ["판단 기록 조회", "log_query", "날짜, 정시", "조치·근거 문장, (사후) 놓친 칸·헛경보", "decision_log.csv"],
], columns=["도구", "함수", "입력", "출력", "원천"]).set_index("도구")


# ================================================================ 의도 분류
CONTROL_WORDS = ["차단해", "꺼 줘", "꺼줘", "끄세요", "꺼라", "켜줘", "켜 줘", "정지시켜", "멈춰", "설정 바꿔", "설정을 바꿔",
                 "기준을 바꿔", "여유값을 바꿔", "자동으로 실행", "확정해", "반영해"]
STAFF_WORDS = ["작업자", "인원", "몇 명", "사람 수", "채용", "근무표", "교대조 편성"]
SCOPE_WORDS = ["고장", "원인 설비", "어느 설비", "날씨", "기온 예보", "다른 공장", "인건비", "주가", "불량", "품질",
               "2022", "내년", "작년", "온도"]

INTENTS = {   # 의도: [(키워드, 점수)]
    "mc_risk":       [("준수율", 3), ("몬테카를로", 3), ("지키면", 2), ("안 지키", 2), ("못 지키", 2), ("절감 0원", 3), ("확률", 1)],
    "tariff":        [("요금", 2), ("얼마", 1), ("기본요금", 3), ("전력량요금", 3), ("돈", 1)],
    "best_block":    [("언제 넣", 3), ("어디에 넣", 3), ("가장 좋", 2), ("제일 좋", 2), ("최선", 2), ("몇 시에 넣", 3), ("언제 돌리", 3), ("언제가 좋", 3)],
    "whatif_block":  [("넣으면", 3), ("돌리면", 3), ("시작하면", 3), ("옮기면", 2), ("하면 어떻게", 2), ("넣을 수 있", 2),
                      ("넣었다면", 3), ("넣었으면", 3), ("돌렸다면", 3)],
    "lp_plan":       [("배치", 2), ("나눠", 2), ("옮길 계획", 3), ("LP", 3), ("최적", 2), ("옮긴 비율", 2), ("% 옮기", 3)],
    "log_query":     [("기록", 3), ("놓친", 3), ("헛경보", 3), ("맞았", 2), ("사후", 2), ("어땠", 2), ("결과", 1), ("실제", 1)],
    "alarm":         [("경보", 2), ("준비", 2), ("알림", 2), ("판단", 1), ("울려", 2), ("위험해", 1), ("조치", 1), ("할 일", 2), ("근거", 1)],
    "forecast_hour": [("1시간 앞", 3), ("한 시간 앞", 3), ("다음 1시간", 3), ("이번 시간", 2)],
    "forecast_day":  [("하루 앞", 3), ("내일", 2), ("위험 시간대", 2), ("예측 최대", 2), ("최대 예측", 2), ("몇 시가 제일", 2), ("예측", 1)],
}
# 필수 근거 단어: 점수가 높아도 이 중 하나가 없으면 답하지 않고 되묻는다 (지난 시험의 '작업자 배치' 오답 대책)
REQUIRED = {
    "mc_risk":       ["준수", "몬테카를로", "지키", "지킨"],
    "tariff":        ["요금", "원", "돈"],
    "best_block":    ["kW", "작업", "열처리", "절단", "부하", "설비"],
    "whatif_block":  ["kW", "작업", "열처리", "절단", "부하", "설비"],
    "lp_plan":       ["옮기", "옮길", "옮긴", "부하", "%", "LP", "받는 칸"],
    "log_query":     ["기록", "놓친", "헛경보", "사후", "결과", "어땠", "맞았"],
    "alarm":         ["경보", "준비", "알림", "판단", "울려", "할 일", "조치"],
    "forecast_hour": ["예측"],
    "forecast_day":  ["예측", "위험 시간대"],
}
LABEL = {"forecast_day": "하루 앞 예측 조회", "forecast_hour": "1시간 앞 예측 조회", "alarm": "경보 판단",
         "whatif_block": "작업을 특정 시각에 넣는 What-if", "best_block": "작업의 가장 좋은 시작 시각",
         "lp_plan": "받는 칸 배치(LP)", "mc_risk": "규칙 준수율별 위험(몬테카를로)", "tariff": "요금 계산", "log_query": "판단 기록 조회"}
EXAMPLE = {"forecast_day": "'9/01 하루 앞 예측 최대는?'", "forecast_hour": "'9/01 8시 1시간 앞 예측은?'",
           "alarm": "'9/01 8시 경보 판단은?'", "whatif_block": "'9/01 열처리 30kW 2시간을 18시에 넣으면?'",
           "best_block": "'9/01 30kW 2시간 작업은 언제 넣는 게 좋아?'", "lp_plan": "'9/01 위험 시간대 부하 10% 옮길 계획 배치해 줘'",
           "mc_risk": "'준수율 90%면 절감은?'", "tariff": "'피크 10kW면 기본요금 1년 얼마?'", "log_query": "'9/01 판단 기록 보여 줘'"}


def classify(q: str):
    """(의도, 상태) — 상태: ok / refused / clarify"""
    if any(k in q for k in CONTROL_WORDS):
        return "refuse_control", "refused"
    if any(k in q for k in STAFF_WORDS):
        return "refuse_staff", "refused"
    if any(k in q for k in SCOPE_WORDS):
        return "refuse_scope", "refused"
    sc = {it: sum(w for k, w in kws if k in q) for it, kws in INTENTS.items()}
    order = list(INTENTS)
    best = max(order, key=lambda it: (sc[it], -order.index(it)))
    if sc[best] < 2:
        return "refuse_unknown", "refused"
    if not any(k.lower() in q.lower() for k in REQUIRED[best]):
        return best, "clarify"
    return best, "ok"


# ================================================================ 슬롯 추출
def extract_slots(q: str, now: pd.Timestamp):
    s = {}
    m = re.search(r"(\d{1,2})\s*/\s*(\d{1,2})", q) or re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", q)
    if m:
        s["date"] = pd.Timestamp(2021, int(m.group(1)), int(m.group(2)))
    elif "내일" in q:
        s["date"] = now.normalize() + pd.Timedelta(days=1)
    else:
        s["date"] = now.normalize()                         # 날짜를 말하지 않으면 오늘(리플레이 시계)
    m = re.search(r"(\d{1,2})\s*시\s*(\d{1,2})\s*분", q) or re.search(r"(\d{1,2}):(\d{2})", q)
    if m:
        s["hour"] = int(m.group(1)) + int(m.group(2)) / 60
    else:
        m = re.search(r"(오후\s*)?(\d{1,2})\s*시(?!간)", q)
        if m:
            h = int(m.group(2)); s["hour"] = h + 12 if m.group(1) and h < 12 else h
    s["hour_explicit"] = "hour" in s
    if not s["hour_explicit"] and s["date"] == now.normalize():
        s["hour"] = now.hour                                # 시각을 말하지 않으면 지금의 정시 (경보·1시간 앞 예측에만 씀)
    m = re.search(r"(\d+(?:\.\d+)?)\s*kW(?!h)", q, re.I)
    if m: s["kw"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*kWh", q, re.I)
    if m: s["kwh"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*시간(?!\s*앞)", q)
    if m and "다음 1시간" not in q: s["dur"] = float(m.group(1))
    m = re.search(r"(\d+)\s*%", q)
    if m: s["pct"] = int(m.group(1))
    s["focus"] = any(k in q for k in ["주의일", "월요일"])
    return s


# ================================================================ 계획 (의도 → 도구 호출 목록, 고정표)
DEFAULT_DUR = 2.0


def make_plan(intent, s, now):
    a = {"as_of": str(now)}; d = s.get("date")
    hx = s.get("hour") if s.get("hour_explicit") else None
    if intent == "forecast_day":
        return [("forecast", dict(date=d, kind="day", hour_=hx))]
    if intent == "forecast_hour":
        return [("forecast", dict(date=d, kind="hour", hour_=s.get("hour"), **a))]
    if intent == "alarm":
        return [("alarm", dict(date=d, hour_=s.get("hour"), **a))]
    if intent == "whatif_block":
        dur = s.get("dur", DEFAULT_DUR)
        calls = [("whatif_block", dict(date=d, start=hx, hours=dur, kw=s.get("kw"), **a))]
        if s.get("kw") is not None and hx is not None:
            calls.append(("tariff", dict(kwh=s["kw"] * dur, at=str(d + pd.Timedelta(hours=hx)))))
        return calls
    if intent == "best_block":
        return [("best_block", dict(date=d, hours=s.get("dur", DEFAULT_DUR), kw=s.get("kw"), **a))]
    if intent == "lp_plan":
        return [("lp_plan", dict(date=d, f=s.get("pct", 10) / 100, **a))]
    if intent == "mc_risk":
        return [("mc_risk", dict(compliance=s.get("pct"), policy="주의일 반드시 준수" if s["focus"] else "균일 준수"))]
    if intent == "tariff":
        if s.get("kwh") is not None:
            return [("tariff", dict(kwh=s["kwh"], at=str(d + pd.Timedelta(hours=hx or 0))))]
        return [("tariff", dict(kw=s.get("kw")))]
    if intent == "log_query":
        return [("log_query", dict(date=d, hour_=int(hx) if hx is not None else None, **a))]
    return []


# ================================================================ 답변 템플릿 (문장 틀에는 숫자를 직접 쓰지 않음, TEMPLATE_LITERALS 제외)
def _won(x): return f"{x:,.0f}원"
def _lst(v): return ", ".join(map(str, v)) if v else "없음"


def render(intent, calls):
    o = calls[0][2]; B = o.get("기준", {})
    if intent == "forecast_day":
        txt = (f"{o['날짜']}({o['범위']}) 하루 앞 예측 최대는 {o['예측최대']}kW({o['예측최대_시각']})입니다. "
               f"준비선({B['준비선']}kW) 이상인 시간은 {_lst([f'{h}시' for h in o['준비선이상_시간']])}입니다.")
        if o["주의일"]: txt += " 전날 쉰 가동일(주의일)이라 오전에는 전력계를 직접 확인하세요."
        return txt + " 하루 앞 예측은 '어느 시간대가 위험한가'를 보는 데만 쓰세요."
    if intent == "forecast_hour":
        cells = " / ".join(f"{k} {v}" for k, v in o["칸별예측"].items())
        return (f"{o['발행시각']} 발행 한 시간 앞 예측: {cells}kW. 최대 {o['예측최대']}kW({o['예측최대_시각']}), "
                f"준비선({B['준비선']}kW)과의 차이 {o['준비선까지']}kW(음수면 준비선 위).")
    if intent == "alarm":
        if o["판단"] == "준비":
            txt = (f"[준비] {o['발행시각']}: 경보 칸 {o['경보칸_수']}개({_lst(o['경보칸_시각'])}), 예측 최대 {o['예측최대']}kW({o['예측최대_시각']}). "
                   f"새 부하를 더하지 말고 대용량 설비 기동은 다음 정시 이후로 미루세요. "
                   f"이런 경보는 그 주 이전 기록으로 {o['N번중1번']}번 중 한 번꼴로 실제 피크였습니다(실제보다 낙관적일 수 있음).")
        else:
            txt = (f"[정상] {o['발행시각']}: 예측 최대 {o['예측최대']}kW({o['예측최대_시각']}), "
                   f"준비선까지 {o['준비선까지']}kW 여유. 평소대로 운영하세요.")
        if o["주의일"]: txt += " 주의일이라 전력계를 직접 확인하세요."
        return txt + f" 2단계({B['2단계']}kW)는 관리장치가 자동 차단합니다."
    if intent == "whatif_block":
        txt = (f"{o['날짜']} {o['시작']}부터 {o['길이_시간']:g}시간 동안 {o['크기_kW']:g}kW(질문에서 준 값)를 더하면, 하루 앞 예측 기준으로 "
               f"그 구간 최대 {o['넣은칸_예측최대']}kW({o['넣은칸_예측최대_시각']}), 준비선을 넘는 칸 {o['준비선_넘는칸']}개, "
               f"1단계를 넘는 칸 {o['1단계_넘는칸']}개입니다. 판정: {o['판정']}.")
        if o["위험시간대_걸림"]: txt += " 위험 시간대에 걸립니다."
        elif o["받는칸_규칙위반칸"]: txt += f" 받는 칸 규칙에 어긋나는 칸이 {o['받는칸_규칙위반칸']}개 있습니다."
        if "사후_넣은칸_실제최대" in o:
            txt += (f" (사후 확인) 실제 전력에 더하면 구간 최대 {o['사후_넣은칸_실제최대']}kW, 1단계 초과 {o['사후_1단계_넘는칸']}칸, "
                    f"여름 최대 {o['여름최대_원래']} → {o['사후_여름최대']}kW.")
        if len(calls) > 1:
            t = calls[1][2]
            txt += f" 이 작업의 전력량요금은 {t['시간대']} 단가 {t['단가_원_kWh']}원/kWh로 {_won(t['전력량요금_원'])}입니다."
        return txt
    if intent == "best_block":
        alt = "; ".join(f"{c['시작']} {c['예측최대']}kW" for c in o["차선"])
        return (f"{o['날짜']}에 {o['크기_kW']:g}kW·{o['길이_시간']:g}시간 작업은 받는 칸 안 후보 {o['후보수']}개 중 "
                f"{o['최선_시작']} 시작이 가장 낫습니다(예측 최대 {o['최선_예측최대']}kW, 준비선까지 {o['최선_준비선까지']}kW 여유). "
                f"다음 후보: {alt}. 가장 나쁜 후보는 {o['최악_시작']}({o['최악_예측최대']}kW). 최종 결정은 생산관리 담당이 합니다.")
    if intent == "lp_plan":
        rec = ", ".join(f"{k} {v}kWh" for k, v in o["받는칸_시간별_kWh"].items())
        txt = (f"{o['날짜']} 위험 시간대 부하의 {round(o['옮긴비율_f'] * 100)}%({o['옮길양_kWh']}kWh)를 LP로 배치하면 "
               f"받는 칸: {rec}. 계획상 하루 최대 {o['계획_하루최대_전']} → {o['계획_하루최대_후']}kW, "
               f"받는 칸 최대 {o['계획_받는칸_최대']}kW, 전력량요금 변화 {_won(o['전력량요금_변화_원'])}.")
        if "사후_하루최대_후" in o:
            txt += f" (사후 확인) 실제 하루 최대 {o['사후_하루최대_전']} → {o['사후_하루최대_후']}kW."
        return txt
    if intent == "mc_risk":
        return (f"{o['정책']}, 준수율 {o['준수율']}%일 때: 기대 절감 연 {o['기대절감_만원']}만 원, 하위 5% 경우 {o['하위5%_절감_만원']}만 원, "
                f"절감이 전혀 없을 확률 {o['절감0원_확률']}%, 여름 최대 중앙값 {o['여름최대_중앙값']}kW(95% 분위 {o['여름최대_95분위']}kW). "
                f"출처: {o['출처']}.")
    if intent == "tariff":
        if "연_기본요금_원" in o:
            return (f"피크 {o['피크_kW']:g}kW는 기본요금으로 1년에 {_won(o['연_기본요금_원'])}입니다"
                    f"(단가 {o['기본요금단가_원_kW월']:,}원/kW·월 × 12개월, {o['계약']}).")
        return f"{o['전력량_kWh']:g}kWh를 {o['시간대']}(단가 {o['단가_원_kWh']}원/kWh)에 쓰면 전력량요금 {_won(o['전력량요금_원'])}입니다."
    if intent == "log_query":
        if "조치문장" in o:
            txt = f"{o['발행시각']} 판단: [{o['판단']}]. 조치: {o['조치문장']} 근거: {o['근거문장']}"
            if "사후_실제최대" in o:
                txt += (f" (사후 확인) 그 시간대 실제 최대 {o['사후_실제최대']}kW, 1단계 초과 {o['사후_1단계초과칸']}칸 중 "
                        f"경보 칸으로 잡은 것 {o['사후_경보로잡은칸']}칸.")
            else:
                txt += " 실제값은 그 시간이 지난 뒤에 보여 드립니다."
            return txt
        txt = f"{o['날짜']} 판단 {o['판단수']}회 중 [준비] {o['준비수']}회({_lst(o['준비_시각'])})."
        if "사후_놓친칸" in o:
            txt += (f" (사후 확인, 지난 판단 {o['사후_대상_판단수']}회) 1단계 초과 {o['사후_1단계초과칸']}칸 중 {o['사후_경보로잡은칸']}칸을 경보 칸으로 잡았고 "
                    f"{o['사후_놓친칸']}칸을 놓쳤습니다(놓친 칸이 있던 시간: {_lst(o['사후_놓친칸이있던_시각'])}). "
                    f"헛경보 시간: {_lst(o['사후_헛경보_시각'])}.")
        return txt
    raise ValueError(intent)


REFUSALS = {
    "refuse_control": "설비를 켜고 끄거나 설정·계획을 바꾸는 일은 이 도우미가 하지 않습니다. 조치는 반장이 판단해 실행하고, 2단계 차단은 최대전력 관리장치가 맡습니다.",
    "refuse_staff": "인원 배치는 이 도우미의 범위가 아닙니다. 이 도우미는 전력 예측·경보·작업(부하) 시각·요금·판단 기록만 다룹니다.",
    "refuse_scope": "이 질문은 도우미가 가진 계산 도구(예측·경보·배치·몬테카를로·요금·판단 기록) 밖이라 답하지 않습니다. 숫자를 지어내지 않기 위해서입니다.",
    "refuse_unknown": "질문을 이해하지 못했습니다. 예: '9/01 8시 경보 판단은?', '9/01 열처리 30kW 2시간을 18시에 넣으면?', '준수율 90%면 절감은?'",
}


# ================================================================ 실행·근거 카드·감사 기록
@dataclass
class Turn:
    question: str; role: str; now: str; intent: str; slots: dict
    calls: list = field(default_factory=list)       # (도구, 입력, 출력)
    answer: str = ""; status: str = "answered"      # answered / refused / clarify / need_info / tool_error
    approval: str = "해당 없음"


def _jsonable(v):
    return str(v) if isinstance(v, pd.Timestamp) else v


def ask(q: str, now="2021-09-01 08:00", role="-") -> Turn:
    now = pd.Timestamp(now)
    intent, state = classify(q)
    s = extract_slots(q, now) if state != "refused" else {}
    turn = Turn(q, role, str(now), intent, {k: _jsonable(v) for k, v in s.items()})
    if state == "refused":
        turn.answer, turn.status = REFUSALS[intent], "refused"
        return turn
    if state == "clarify":
        turn.answer = (f"질문을 '{LABEL[intent]}'로 읽었지만 그렇게 볼 근거 단어가 부족해 답하지 않습니다. "
                       f"그 뜻이 맞다면 이렇게 물어 주세요: {EXAMPLE[intent]}")
        turn.status = "clarify"
        return turn
    try:
        for name, kw in make_plan(intent, s, now):
            out = TOOLS[name](**kw)
            turn.calls.append((name, {k: _jsonable(v) for k, v in kw.items()}, {k: v for k, v in out.items() if not k.startswith("_")}))
        turn.answer = f"[이해한 질문: {LABEL[intent]}] " + render(intent, turn.calls)
        if intent in ("whatif_block", "best_block", "lp_plan"):
            turn.approval = "제안만 함 — 생산관리 담당 승인 전에는 계획표에 반영하지 않음"
    except ToolError as e:
        turn.calls = []
        turn.answer = f"답할 수 없습니다: {e}."
        turn.status = "need_info" if "필요" in str(e) else "tool_error"
    return turn


def evidence(turn: Turn) -> str:
    """답변 아래에 붙이는 근거 카드 (도구명, 입력, 출력)"""
    if not turn.calls:
        return "  (도구 호출 없음)"
    lines = []
    for name, inp, out in turn.calls:
        inp = {k: v for k, v in inp.items() if v is not None}
        brief = {k: v for k, v in out.items() if k != "기준"}
        lines.append(f"  도구 {name} | 입력 {json.dumps(inp, ensure_ascii=False)}\n    출력 {json.dumps(brief, ensure_ascii=False)}")
    return "\n".join(lines)


def audit_record(turn: Turn) -> dict:
    return {"지금시각": turn.now, "역할": turn.role, "질문": turn.question, "의도": turn.intent, "슬롯": turn.slots,
            "도구호출": [{"도구": n, "입력": i, "출력": o} for n, i, o in turn.calls],
            "답변": turn.answer, "상태": turn.status, "승인": turn.approval}


# ================================================================ 숫자 근거 점검 (답변의 숫자가 모두 도구 입출력에 있는가)
TEMPLATE_LITERALS = [r"[12]단계", r"95% 분위", r"하위 5%", r"× 12개월", r"선택Ⅰ", r"2021년", r"1년"]


def _flatten(x, acc):
    if isinstance(x, dict):
        for v in x.values(): _flatten(v, acc)
    elif isinstance(x, (list, tuple)):
        for v in x: _flatten(v, acc)
    elif isinstance(x, bool):
        pass
    elif isinstance(x, (int, float, np.integer, np.floating)):
        acc.add(round(float(x), 1)); acc.add(round(float(x) * 100, 1))      # 비율 → % 표기
    elif isinstance(x, str):
        for m in re.findall(r"-?\d+(?:\.\d+)?", x.replace(",", "")): acc.add(round(float(m), 1))
    return acc


def number_check(turn: Turn):
    """(답변 속 숫자·시각 표기 수, 근거가 있는 수, 근거 없는 목록)"""
    allowed = set()
    for _, i, o in turn.calls:
        _flatten(i, allowed); _flatten(o, allowed)
    text = turn.answer.replace(",", "")
    blob = json.dumps([c[1:] for c in turn.calls], ensure_ascii=False, default=str)
    times = re.findall(r"\d{1,2}/\d{2}|\d{1,2}:\d{2}|\d+시(?!간)", text)
    bad_t = [t for t in times if t not in blob and t.replace("시", "") not in blob]
    for p in [r"\d{1,2}/\d{2}", r"\d{1,2}:\d{2}", r"\d+시(?!간)"] + TEMPLATE_LITERALS:
        text = re.sub(p, " ", text)
    nums = [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", text)]
    bad = [n for n in nums if round(n, 1) not in allowed]
    n = len(nums) + len(times)
    return n, n - len(bad) - len(bad_t), bad + bad_t


# ================================================================ 평가셋
NON_ANSWER = {"refused", "clarify", "need_info", "tool_error"}


def _dlog_judgement(day, h):
    return data().dlog.set_index("발행시각").loc[pd.Timestamp(day) + pd.Timedelta(hours=h), "판단"]


def _chk_alarm(day, h):
    return lambda t: t.status == "answered" and t.calls[0][2]["판단"] == _dlog_judgement(day, h)


def _v(key, val):
    return lambda t: t.status == "answered" and t.calls[0][2].get(key) == val


def _st(*status):
    return lambda t: t.status in status


# (역할, 지금 시각, 질문, 기대 의도, 기대 도구, 과제 성공 점검)
SET_A = [   # 개발용: 이 세트를 보며 규칙을 만들고 고침
    ("반장", "2021-09-01 08:00", "지금 경보 상황이야? 준비해야 해?", "alarm", ["alarm"], _chk_alarm("2021-09-01", 8)),
    ("반장", "2021-09-01 08:00", "다음 1시간 예측 보여줘", "forecast_hour", ["forecast"], _v("예측최대", 181.3)),
    ("반장", "2021-09-01 10:00", "9/01 10시 경보 판단과 할 일은?", "alarm", ["alarm"], _chk_alarm("2021-09-01", 10)),
    ("반장", "2021-09-01 08:00", "8시 판단 기록 보여 주고 실제로 어땠는지도 알려줘", "log_query", ["log_query"],
     lambda t: "사후_실제최대" not in t.calls[0][2] and "지난 뒤에" in t.answer),
    ("반장", "2021-09-01 11:00", "9/01 11시 알림 울려?", "alarm", ["alarm"], _chk_alarm("2021-09-01", 11)),
    ("반장", "2021-09-01 08:00", "압축기 지금 꺼줘", "refuse_control", [], _st("refused")),
    ("생산관리", "2021-08-31 16:00", "내일 하루 앞 예측 최대랑 위험 시간대 알려줘", "forecast_day", ["forecast"],
     lambda t: t.calls[0][2]["예측최대"] == 189.7 and t.calls[0][2]["날짜"] == "09/01"),
    ("생산관리", "2021-08-31 16:00", "9/01 열처리 30kW 2시간을 9시에 넣으면 어떻게 돼?", "whatif_block", ["whatif_block", "tariff"],
     lambda t: t.calls[0][2]["판정"] == "권하지 않음" and "사후_여름최대" not in t.calls[0][2]),
    ("생산관리", "2021-08-31 16:00", "9/01 열처리 30kW 2시간을 18시에 넣으면?", "whatif_block", ["whatif_block", "tariff"], _v("넣은칸_예측최대", 194.2)),
    ("생산관리", "2021-08-31 16:00", "9/01에 열처리 30kW 2시간은 언제 넣는 게 가장 좋아?", "best_block", ["best_block"], _v("최선_시작", "20:00")),
    ("생산관리", "2021-08-31 16:00", "9/01 위험 시간대 부하 10% 옮길 계획 배치해 줘", "lp_plan", ["lp_plan"],
     lambda t: t.calls[0][2]["계획_하루최대_후"] < t.calls[0][2]["계획_하루최대_전"]),
    ("생산관리", "2021-08-31 16:00", "9/01 열처리를 18시에 넣으면 괜찮아?", "whatif_block", [], _st("need_info")),
    ("에너지", "2021-09-06 09:00", "9/01 판단 기록에서 놓친 피크랑 헛경보 정리해줘", "log_query", ["log_query"], _v("사후_놓친칸", 1)),
    ("에너지", "2021-09-06 09:00", "규칙 준수율 90%면 절감은 얼마나 돼?", "mc_risk", ["mc_risk"], _v("기대절감_만원", 84.2)),
    ("에너지", "2021-09-06 09:00", "주의일은 반드시 지키고 준수율 80%면 절감 0원 확률은?", "mc_risk", ["mc_risk"], _v("절감0원_확률", 0.0)),
    ("에너지", "2021-09-06 09:00", "피크 18.5kW 줄이면 기본요금 1년에 얼마 아껴?", "tariff", ["tariff"], _v("연_기본요금_원", 1602840)),
    ("에너지", "2021-09-06 09:00", "준수율 85%면 기대 절감은?", "mc_risk", [], _st("tool_error")),
    ("에너지", "2021-09-06 09:00", "9/01 열처리 30kW 2시간을 9시에 넣었다면 여름 최대는?", "whatif_block", ["whatif_block", "tariff"], _v("사후_여름최대", 226.0)),
    ("에너지", "2021-09-06 09:00", "8월 피크를 만든 고장 설비가 뭐야?", "refuse_scope", [], _st("refused")),
    ("생산관리", "2021-08-31 16:00", "내일 날씨 어때?", "refuse_scope", [], _st("refused")),
]

SET_B = [   # 지난 시험에서 세트 A로 규칙을 정한 뒤 새로 쓴 질문 (그때 12/14). 실패 2건은 이번에 고침
    ("반장", "2021-09-01 13:00", "이번 시간 준비 단계야?", "alarm", ["alarm"], _chk_alarm("2021-09-01", 13)),
    ("반장", "2021-09-01 14:00", "14시 한 시간 앞 예측값 칸별로", "forecast_hour", ["forecast"], _v("예측최대", 184.4)),
    ("반장", "2021-09-01 15:00", "방금 15시 판단 근거가 뭐야", "alarm", ["alarm"], _chk_alarm("2021-09-01", 15)),
    ("반장", "2021-09-01 09:00", "공조 설정 바꿔줘", "refuse_control", [], _st("refused")),
    ("생산관리", "2021-09-02 16:00", "9월 3일 최대 예측 몇 시가 제일 높아?", "forecast_day", ["forecast"], _v("날짜", "09/03")),
    ("생산관리", "2021-09-02 16:00", "9/03 소재 절단 20kW 1시간을 오후 7시에 돌리면?", "whatif_block", ["whatif_block", "tariff"], _v("시작", "19:00")),
    ("생산관리", "2021-09-02 16:00", "9/03 25kW짜리 2시간 작업 몇 시에 넣을까 제일 좋은 시각", "best_block", ["best_block"], _v("크기_kW", 25.0)),
    ("생산관리", "2021-09-02 16:00", "9/03에 15% 옮기면 받는 칸 어떻게 나눠?", "lp_plan", ["lp_plan"], _v("옮긴비율_f", 0.15)),
    ("에너지", "2021-09-13 09:00", "9/08 경보 결과 어땠어? 놓친 거 있어?", "log_query", ["log_query"], _v("날짜", "09/08")),
    ("에너지", "2021-09-13 09:00", "규칙을 95% 지키면 몬테카를로로 절감 얼마?", "mc_risk", ["mc_risk"], _v("기대절감_만원", 103.1)),
    ("에너지", "2021-09-13 09:00", "10kW 넘으면 기본요금 1년 얼마야", "tariff", ["tariff"], _v("연_기본요금_원", 866400)),
    ("에너지", "2021-09-13 09:00", "8/20 9시 판단 기록", "log_query", ["log_query"], _v("판단", "정상")),
    ("에너지", "2021-09-13 09:00", "품질 불량률은 어때?", "refuse_scope", [], _st("refused")),
    ("생산관리", "2021-09-13 09:00", "작업자 몇 명 배치해야 돼?", "refuse_staff", [],
     lambda t: t.status == "refused" and "인원 배치는 이 도우미의 범위가 아닙니다" in t.answer),
]

SET_C = [   # 이번 수정(필수 근거 단어·인원 거절·되돌려 말하기)을 마친 뒤 새로 쓴 함정 질문. 돌린 뒤 규칙을 고치지 않음
    ("생산관리", "2021-09-02 16:00", "내일 야간조 인원 몇 명 넣어야 해?", "refuse_staff", [], _st("refused")),
    ("생산관리", "2021-09-02 16:00", "9/03 배치 끝났어?", "lp_plan", [], _st("clarify")),
    ("반장", "2021-09-01 08:00", "LP 설정 바꿔서 다시 돌려", "refuse_control", [], _st("refused")),
    ("반장", "2021-09-01 08:00", "열처리로 몇 도로 맞춰?", "refuse_unknown", [], _st("refused")),
    ("반장", "2021-09-20 08:00", "9/20 8시 경보 판단은?", "alarm", [], _st("tool_error")),
    ("에너지", "2021-09-06 09:00", "준수율 70%면 절감 얼마야?", "mc_risk", [], _st("tool_error")),
    ("에너지", "2021-09-06 09:00", "9/01 요금 얼마 나왔어?", "tariff", [], _st("need_info")),
    ("반장", "2021-09-01 08:00", "9/01 9시 판단 기록이랑 실제 결과", "log_query", [], _st("tool_error")),
    ("반장", "2021-09-01 08:00", "10시 경보 미리 알려줘", "alarm", [], _st("tool_error")),
    ("생산관리", "2021-08-31 16:00", "9/01 배치 계획 확정해", "refuse_control", [], _st("refused")),
    ("생산관리", "2021-08-31 16:00", "9/01 20kW 1시간 작업을 오후 2시에 넣으면?", "whatif_block", ["whatif_block", "tariff"],
     lambda t: t.calls[0][2]["위험시간대_걸림"] and "사후_여름최대" not in t.calls[0][2]),
    ("에너지", "2021-09-06 09:00", "결과가 좋았던 이유가 뭐야?", "refuse_unknown", [], _st("clarify", "refused")),
]

# 지난 시험에서 틀린 질문 (세트 B), 그때의 결과와 이번 결과
PREV_FAILURES = [
    ("생산관리", "2021-09-13 09:00", "작업자 몇 명 배치해야 돼?", "받는 칸 배치(LP) 계산을 답함 (오답, '배치' 한 단어로 오분류)"),
    ("반장", "2021-09-01 15:00", "방금 15시 판단 근거가 뭐야", "이해하지 못했다며 거절 ('근거'가 키워드에 없음)"),
]

SETS = {"A": SET_A, "B": SET_B, "C": SET_C}


def evaluate(name, S):
    """세트 하나의 지표 dict와 문항별 표"""
    rows = []; ok_int = ok_tool = succ = nums = good = exp_non = non_ok = same = 0
    for role, now, q, exp_int, exp_tools, chk in S:
        t = ask(q, now, role); t2 = ask(q, now, role)
        same += json.dumps(audit_record(t), ensure_ascii=False, default=str) == json.dumps(audit_record(t2), ensure_ascii=False, default=str)
        ok_int += t.intent == exp_int
        ok_tool += [c[0] for c in t.calls] == exp_tools
        try:
            s_ok = bool(chk(t))
        except Exception:
            s_ok = False
        succ += s_ok
        if not exp_tools:                     # 답하지 않아야 하는 문항 (거절·되묻기·입력 부족·기간 밖)
            exp_non += 1; non_ok += t.status in NON_ANSWER
        if t.status == "answered":
            a, b, _ = number_check(t); nums += a; good += b
        rows.append({"세트": name, "역할": role, "지금 시각": now[5:], "질문": q, "기대 의도": exp_int, "의도": t.intent,
                     "도구": ", ".join(c[0] for c in t.calls) or "-", "상태": t.status, "성공": s_ok})
    n = len(S)
    res = {"질문 수": n, "의도 정확도": f"{ok_int}/{n}", "도구 선택 정확도": f"{ok_tool}/{n}",
           "답하지 않아야 할 질문을 답하지 않음": f"{non_ok}/{exp_non}", "과제 성공": f"{succ}/{n}",
           "숫자 근거 일치": f"{good}/{nums}", "반복 실행 동일": f"{same}/{n}"}
    return res, pd.DataFrame(rows)
