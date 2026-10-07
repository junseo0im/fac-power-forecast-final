"""계산 도구를 골라 쓰는 규칙 기반 질의응답 도우미(notebooks/peak_helper.py) 점검.

실행: 노트북을 모두 실행한 뒤 `python -m pytest tests -q`
데이터나 예측 파일이 없으면 건너뜁니다.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import peak_helper as pa  # noqa: E402

TAB = ROOT / "outputs" / "tables"


@pytest.fixture(scope="module")
def D():
    if not pa.files_ready():
        pytest.skip("전처리 데이터·예측·판단 기록이 없음 — 노트북 01-1~04-2를 먼저 실행하세요")
    return pa.data()


def need(path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} 없음")


def all_questions():
    return [row for S in pa.SETS.values() for row in S]


def test_경보도구는_판단기록과_모두_같다(D):
    """04-2 노트북 decision_log의 모든 정시 판단을 경보 도구가 똑같이 다시 낸다"""
    for t, row in D.dlog.set_index("발행시각").iterrows():
        out = pa.alarm(t.normalize(), t.hour)
        assert out["판단"] == row["판단"] and out["판단기록과_일치"], t
        assert out["경보칸_수"] == row["경보칸_수"]


def test_표40_30_42를_재현한다(D):
    """LP 도구 = 표 40 '하루 앞 LP (최종 모델)', 요금 도구 = 표 30, 몬테카를로 도구 = 표 42"""
    for p in ["t40_MPC_비교.csv", "t30_요금_절감추정.csv", "t42_몬테카를로_강건성.csv"]:
        need(TAB / p)
    t40 = pd.read_csv(TAB / "t40_MPC_비교.csv", encoding="utf-8-sig")
    days = [d for d in D.daily.index if pa.FIRST_TEST <= d < pa.TEST_END and D.daily.loc[d, "하루종류"] == "평일가동"]
    for f in [0.10, 0.15]:
        peak = max(pa.lp_plan(d, f, amount_from="actual", observed_only=True, return_series=True)["_series"].max() for d in days)
        exp = t40[(t40["옮긴 비율 f"] == f) & (t40["방안"] == "하루 앞 LP (최종 모델)")]["조정 후 최대수요전력"].iloc[0]
        assert round(peak, 1) == exp, (f, peak, exp)
    t30 = pd.read_csv(TAB / "t30_요금_절감추정.csv", encoding="utf-8-sig", index_col=0).iloc[0]
    assert pa.tariff(kw=t30["줄어든 여름 최대(kW)"])["연_기본요금_원"] == t30["연 기본요금 절감(원)"]
    t42 = pd.read_csv(TAB / "t42_몬테카를로_강건성.csv", encoding="utf-8-sig")
    for _, r in t42[t42["옮길 수 있는 부하"].str.startswith("f 5~15%")].iterrows():
        o = pa.mc_risk(r["준수율(%)"], r["정책"])
        assert (o["기대절감_만원"], o["절감0원_확률"], o["여름최대_95분위"]) == \
               (r["기대 절감(만 원/년)"], r["절감 0원 확률(%)"], r["여름 최대 95% 분위(kW)"])


def test_답의_숫자는_모두_도구_출력에서_나온다(D):
    for role, now, q, *_ in all_questions():
        t = pa.ask(q, now, role)
        if t.status == "answered":
            n, ok, bad = pa.number_check(t)
            assert n > 0 and not bad, (q, bad)
        else:
            assert not t.calls, q          # 답하지 않은 질문은 도구 결과를 남기지 않음


def test_미래_정보를_돌려주지_않는다(D):
    """지금 시각 이후의 실제값은 출력에 없고, 그 실제값을 바꿔도 답이 같다"""
    now = pd.Timestamp("2021-09-01 08:00")
    plan_now = "2021-08-31 16:00"
    w = pa.whatif_block("2021-09-01", 9, 2, 30, as_of=plan_now)
    lp = pa.lp_plan("2021-09-01", 0.10, as_of=plan_now)
    lg = pa.log_query("2021-09-01", 8, as_of=now)
    assert not any(k.startswith("사후") and k != "사후" for o in (w, lp, lg) for k in o)
    with pytest.raises(pa.ToolError):
        pa.alarm("2021-09-01", 10, as_of=now)          # 아직 오지 않은 정시
    with pytest.raises(pa.ToolError):
        pa.forecast("2021-09-01", "hour", 9, as_of=now)
    qs = [("8시 판단 기록 보여 주고 실제로 어땠는지도 알려줘", str(now)), ("지금 경보 상황이야? 준비해야 해?", str(now)),
          ("9/01 열처리 30kW 2시간을 9시에 넣으면 어떻게 돼?", plan_now), ("9/01 판단 기록 보여 줘", str(now))]
    before = [pa.ask(q, n).answer for q, n in qs]
    s_orig, l_orig = D.slots["전력"].copy(), D.dlog[["실제_최대", "실제_1단계초과칸", "실제_경보로잡은칸"]].copy()
    try:
        D.slots.loc[D.slots.index >= now, "전력"] += 50          # 미래의 실제값을 크게 바꿈
        fut = D.dlog["발행시각"] >= now
        D.dlog.loc[fut, ["실제_최대", "실제_1단계초과칸", "실제_경보로잡은칸"]] = [999.0, 9, 9]
        assert [pa.ask(q, n).answer for q, n in qs] == before
    finally:
        D.slots["전력"] = s_orig
        D.dlog[["실제_최대", "실제_1단계초과칸", "실제_경보로잡은칸"]] = l_orig


def test_제어_범위밖_함정_질문은_답하지_않는다(D):
    t = pa.ask("작업자 몇 명 배치해야 돼?", "2021-09-13 09:00")          # 처음 시제품의 오답 사례
    assert t.status == "refused" and not t.calls and "인원 배치는 이 도우미의 범위가 아닙니다" in t.answer
    for q in ["압축기 지금 꺼줘", "9/01 배치 계획 확정해", "LP 설정 바꿔서 다시 돌려", "내일 날씨 어때?", "8월 피크를 만든 고장 설비가 뭐야?"]:
        t = pa.ask(q)
        assert t.status == "refused" and not t.calls, q
    t = pa.ask("9/03 배치 끝났어?", "2021-09-02 16:00")              # 근거 단어 부족 → 되묻기
    assert t.status == "clarify" and not t.calls
    for q, now, st in [("9/01 열처리를 18시에 넣으면 괜찮아?", "2021-08-31 16:00", "need_info"),
                       ("9/01 요금 얼마 나왔어?", "2021-09-06 09:00", "need_info"),
                       ("준수율 85%면 기대 절감은?", "2021-09-06 09:00", "tool_error"),
                       ("9/20 8시 경보 판단은?", "2021-09-20 08:00", "tool_error")]:
        assert pa.ask(q, now).status == st, q
    for name, S in pa.SETS.items():                                     # 평가셋 전체: 의도·도구·과제 점검
        res, df = pa.evaluate(name, S)
        assert df["성공"].all() and (df["의도"] == df["기대 의도"]).all(), df[~df["성공"]]


def test_같은_질문은_같은_결과(D):
    for role, now, q, *_ in all_questions():
        a = json.dumps(pa.audit_record(pa.ask(q, now, role)), ensure_ascii=False, default=str)
        b = json.dumps(pa.audit_record(pa.ask(q, now, role)), ensure_ascii=False, default=str)
        assert a == b, q


def test_블라인드_사용자경로와_금지어가_없다():
    """모듈·테스트·실행 스크립트·README·모든 노트북(부록 포함, 실행 출력 포함)·새 표에
    사용자 이름·사용자 폴더 경로·금지어(레포 밖 blind_words.txt)가 없다"""
    home = Path.home()
    generic = {"user", "admin", "owner", "pc", "home"}   # 흔한 계정 이름은 일반 글자와 겹치므로 폴더 경로로만 검사
    blocked = [str(home.parent).lower(), home.parent.as_posix().lower(),
               str(home.parent).replace("\\", "\\\\").lower()]   # 노트북(JSON) 안에서는 역슬래시가 두 번 적힘
    if home.name.lower() not in generic:
        blocked.append(home.name.lower())
    words = ROOT / "blind_words.txt"           # 저장소에 올리지 않는 파일. 있을 때만 읽음
    if words.exists():
        blocked += [w.strip().lower() for w in words.read_text(encoding="utf-8-sig").splitlines() if w.strip()]
    files = [ROOT / "notebooks" / "peak_helper.py", ROOT / "notebooks" / "hourly_utils.py", Path(__file__),
             ROOT / "run_all.py", ROOT / "README.md", TAB / "t51_질의응답_도우미_평가.csv"]
    files += sorted((ROOT / "notebooks").glob("*.ipynb")) + sorted((ROOT / "appendix" / "hourly_max_pipeline" / "notebooks").glob("*.ipynb"))
    for f in [f for f in files if f.exists()]:
        text = f.read_text(encoding="utf-8-sig", errors="ignore").lower()
        hits = [w for w in blocked if w and w in text]
        assert not hits, f"{f.name}: {hits}"
