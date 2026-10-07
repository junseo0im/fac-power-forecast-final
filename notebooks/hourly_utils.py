"""시간 최대 분석 공통 함수 — 15분 분석(노트북 01-2·02-2)과 같은 데이터·평가 기준에서 시간 최대 모델을 평가한다.

공통 기준(15분 분석과 같음)
- 데이터: 2021-07-01 이후만 쓴다(1~6월은 다른 날과 똑같은 복사 기록이 87%).
  7/13·7/15 전력은 결측(행 순서가 시각 순서가 아님), 0값(계측 중단)은 결측, 7/28·7/30은 정답으로 쓰지 않는다.
- 피크: 1단계 188.7kW(학습 기간 7/1~8/8 최대 222kW × 0.85), 준비선 178.7kW(노트북 03-1에서 1주차로 정한 여유값 10kW).
- 평가: 8/9~9/14를 1주씩 5번. 매주 그 이전 기간만으로 학습한다. 경보 문턱·확률 보정은 앞 주들로 정하고 2~5주차에서 평가한다.
- 비교 단위: 시간. 정시에 발행해 그 시간의 15분 4칸을 예측하는 15분 모델의 1시간 앞 예측과, 정시에 그 시간 최대를 예측하는
  시간 최대 모델(트랙 A)은 쓰는 정보 시점이 같다. 시간 최대 = 4칸 중 최댓값, 시간 피크 = 4칸 중 하나라도 188.7kW 이상.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]                 # 통합본 루트
BASE = ROOT                                                # 15분 분석의 데이터·예측(data/processed, outputs/predictions)
KIT = ROOT / 'appendix' / 'hourly_max_pipeline'            # 시간 최대 파이프라인(power_utils, 부록)
OUT = ROOT / 'outputs'
os.environ.setdefault('N_THREADS', '2')
for p in (str(KIT), str(KIT / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)
import power_utils as pu  # noqa: E402  (load_data, make_features, check_no_leak, best_threshold, peak_prob_from_quantiles)

TARGET = 222.0
THETA = round(0.85 * TARGET, 1)                            # 188.7kW
READY = round(THETA - 10, 1)                               # 178.7kW
START = pd.Timestamp('2021-07-01')
BROKEN_DAYS = pd.to_datetime(['2021-07-13', '2021-07-15'])
WEEKS = pd.DataFrame({'week': ['1주차', '2주차', '3주차', '4주차', '5주차'],
                      'start': pd.to_datetime(['2021-08-09', '2021-08-16', '2021-08-23', '2021-08-30', '2021-09-06']),
                      'end': pd.to_datetime(['2021-08-15 23:00', '2021-08-22 23:00', '2021-08-29 23:00',
                                             '2021-09-05 23:00', '2021-09-14 23:00'])})
EVAL_WEEKS = ['2주차', '3주차', '4주차', '5주차']
CLF_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8,
                  colsample_bytree=0.8)                    # STEP 6 피크 분류기와 같은 설정
DAYTYPE = ['dt_op', 'dt_sat', 'dt_off', 'prev_op']


def save_table(df, name, index=False):
    path = OUT / 'tables' / f'{name}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding='utf-8-sig')
    return path


def save_fig(fig, name):
    path = OUT / 'figures' / f'{name}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    return path


def week_of(index):
    """시각마다 시험 주 이름(시험 기간 밖이면 '학습')."""
    ts = pd.DatetimeIndex(index)
    w = np.full(len(ts), '학습', dtype=object)
    for r in WEEKS.itertuples():
        w[(ts >= r.start) & (ts <= r.end)] = r.week
    return w


def load_friend():
    """15분 분석(노트북 01-2·02-2)이 만든 15분 데이터와 최종 예측을 읽는다."""
    rd = lambda p: pd.read_csv(BASE / p, encoding='utf-8-sig')
    slots = rd('data/processed/slots_15min.csv').assign(ts=lambda d: pd.to_datetime(d['ts']))
    days = rd('data/processed/days.csv').assign(date=lambda d: pd.to_datetime(d['date']))
    fh = rd('outputs/predictions/final_hour_ahead.csv')
    fh[['발행시각', '대상시각']] = fh[['발행시각', '대상시각']].apply(pd.to_datetime)
    ph = rd('outputs/predictions/pred_hour_ahead.csv')
    ph[['발행시각', '대상시각']] = ph[['발행시각', '대상시각']].apply(pd.to_datetime)
    fd = rd('outputs/predictions/final_day_ahead.csv').assign(ts=lambda d: pd.to_datetime(d['ts']))
    return {'slots': slots, 'days': days, 'final_hour': fh, 'pred_hour': ph, 'final_day': fd}


def hourly_truth(slots, days):
    """시간 단위 정답: 4칸이 모두 정답으로 쓸 수 있을 때만 시간 최대를 만든다."""
    s = slots.assign(h=slots['ts'].dt.floor('h'))
    ok = s['정답사용'].astype(bool) & s['전력'].notna()
    valid = ok.groupby(s['h']).all()
    y = s.groupby('h')['전력'].max().where(valid)
    t = pd.DataFrame({'y': y, 'valid': valid})
    t.index.name = 'ts'
    t['peak'] = t['y'] >= THETA
    t['date'] = t.index.normalize()
    t['hour'] = t.index.hour
    d = days.set_index('date')
    for c in ['가동일', '하루종류', '전날가동']:
        t[c] = t['date'].map(d[c]).to_numpy()
    t['week'] = week_of(t.index)
    return t


def friend_hour_max(df, col, key='발행시각'):
    """15분 모델의 예측(정시 발행, 4칸)을 시간 최대로 바꾼다."""
    return df.groupby(key)[col].max()


def our_frame(days):
    """시간 단위 표(power_utils)에 공통 데이터 규칙을 적용하고, 01-2의 하루 종류(전날 알 수 있는 가동 계획)를 더한다."""
    df, _ = pu.load_data()
    m = df.copy()
    mask = (m.index < START) | m['date'].isin(BROKEN_DAYS)
    for c in ['y', 'y_obs', 'usage'] + list(pu.QUARTER_COLS):
        m.loc[mask, c] = np.nan
    m['op_day'] = m['op_day'].astype(float)
    m.loc[m.index < START, 'op_day'] = np.nan
    m.loc[m['date'].isin(BROKEN_DAYS), 'op_day'] = 1.0           # 01-2: 두 날 모두 평일 가동일(최대 190·202kW)
    d = days.set_index('date')
    m['dt_op'] = m['date'].map(d['하루종류'].eq('평일가동')).astype(float)
    m['dt_sat'] = m['date'].map(d['하루종류'].eq('토요일')).astype(float)
    m['dt_off'] = m['date'].map(d['하루종류'].eq('휴일')).astype(float)
    m['prev_op'] = m['date'].map(d['전날가동']).astype(float)
    return m


def features(m, track):
    """시간 최대 피처(트랙·S1) + 하루 종류. 7/1 이후 행만 돌려준다."""
    X, _ = pu.make_features(m, track, 'S1')
    return X.join(m[DAYTYPE]).loc[START:]


def train_classifier(X, lab, tr, te, seeds):
    """STEP 6과 같은 가중 LightGBM 피크 분류기(seed 평균 확률)."""
    import lightgbm as lgb
    y = lab.loc[tr].astype(int)
    spw = (len(y) - y.sum()) / max(y.sum(), 1)
    ps = []
    for s in seeds:
        mdl = lgb.LGBMClassifier(**CLF_PARAMS, scale_pos_weight=spw, subsample_freq=1, random_state=s,
                                 n_jobs=pu.N_THREADS, deterministic=True, force_row_wise=True, verbose=-1)
        mdl.fit(X.loc[tr], y)
        ps.append(mdl.predict_proba(X.loc[te])[:, 1])
    return pd.Series(np.mean(ps, axis=0), index=te)


def tuned_lgbm_params(track):
    """부록 파이프라인 STEP 3에서 Optuna로 고른 LightGBM(S1) 설정을 그대로 가져온다(다시 튜닝하지 않음)."""
    import json
    t = pd.read_csv(KIT / 'outputs' / 'tables' / 'tree_best_params.csv', encoding='utf-8-sig')
    r = t[(t['model'] == 'lgbm') & (t['track'] == track) & (t['scenario'] == 'S1')].iloc[0]
    return json.loads(r['params'])


def train_regressor(X, y, tr, te, params, ref_time, seeds):
    w = pu.sample_weight(tr, ref_time, tau=params.get('tau', 'inf'))
    ps = []
    for s in seeds:
        mdl = pu.make_tree_model('lgbm', params, s)
        mdl.fit(X.loc[tr], y.loc[tr], sample_weight=w)
        ps.append(mdl.predict(X.loc[te]))
    return pd.Series(np.mean(ps, axis=0), index=te)


def cls_row(lab, alarm, score=None, prob=None, n_days=None):
    """시간 단위 경보 지표."""
    from sklearn.metrics import average_precision_score
    lab, alarm = np.asarray(lab, bool), np.asarray(alarm, bool)
    tp, fp, fn = int((lab & alarm).sum()), int((~lab & alarm).sum()), int((lab & ~alarm).sum())
    r = {'피크 시간': int(lab.sum()), '경보 시간': int(alarm.sum()), '맞힌 피크': tp, '헛경보': fp, '놓친 피크': fn,
         '재현율': tp / (tp + fn) if tp + fn else np.nan, '정밀도': tp / (tp + fp) if tp + fp else np.nan,
         'F1': 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
         'PR-AUC': average_precision_score(lab, score) if score is not None else np.nan,
         'Brier': float(np.mean((np.asarray(prob) - lab) ** 2)) if prob is not None else np.nan}
    if n_days:
        r['가동일 하루 경보 시간'] = alarm.sum() / n_days
    return r


def boot_diff(dates, stat_fn, n_boot=2000, seed=42):
    """날짜 단위로 다시 뽑아 stat_fn(인덱스 배열)의 95% 신뢰구간을 구한다."""
    days = np.unique(dates)
    groups = {d: np.where(dates == d)[0] for d in days}
    rng = np.random.default_rng(seed)
    vals = [stat_fn(np.concatenate([groups[d] for d in rng.choice(days, len(days))])) for _ in range(n_boot)]
    return np.nanpercentile(vals, [2.5, 97.5])


def f1_of(lab, alarm):
    tp, fp, fn = (lab & alarm).sum(), (~lab & alarm).sum(), (lab & ~alarm).sum()
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan


# --- Chronos-2(시간 최대, 7/1 이후 문맥만) ---------------------------------------------------------
KNOWN_COV = ['hour_sin', 'hour_cos'] + DAYTYPE


MODEL_ID = {'base': 'amazon/chronos-2', 'small': 'autogluon/chronos-2-small'}


def chronos_pipe(size='base'):
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(MODEL_ID[size], device_map='cpu')


def known_cov(m):
    """미래에도 아는 값(시각 sin·cos, 하루 종류·전날 가동). 결측은 0."""
    h = m.index.hour.to_numpy()
    return pd.DataFrame({'hour_sin': np.sin(2 * np.pi * h / 24), 'hour_cos': np.cos(2 * np.pi * h / 24)},
                        index=m.index).join(m[DAYTYPE]).fillna(0.0)


def _quantile_frame(qs, idx):
    q = np.concatenate([qi[0].numpy() for qi in qs])
    out = pd.DataFrame(np.sort(q, axis=1), index=idx, columns=pu.Q_COLS)
    out['median'] = out['q50']
    return out


def chronos_track_a(pipe, m, hours, ctx=1344):
    """정시마다 직전 시간까지(7/1 이후)의 시간 최대 계열로 그 시간 최대를 예측(단변량 zero-shot)."""
    import torch
    y = m['y_obs'].to_numpy(float)
    start = m.index.get_loc(START)
    pos = m.index.get_indexer(hours)
    items = [torch.tensor(y[max(start, o - ctx):o], dtype=torch.float32) for o in pos]
    qs, _ = pipe.predict_quantiles(items, prediction_length=1, quantile_levels=pu.QUANTILES, batch_size=256)
    return _quantile_frame(qs, pd.DatetimeIndex(hours))


def chronos_track_b(pipe, m, days, ctx=1344):
    """매일 00시에 전날까지의 계열 + 미래에도 아는 값(시각, 하루 종류)으로 그날 24시간 최대를 예측."""
    cov = known_cov(m)
    y = m['y_obs'].to_numpy(float)
    start = m.index.get_loc(START)
    pos = m.index.get_indexer(pd.DatetimeIndex(days))
    items = [{'target': y[max(start, o - ctx):o],
              'past_covariates': {c: cov[c].to_numpy(float)[max(start, o - ctx):o] for c in KNOWN_COV},
              'future_covariates': {c: cov[c].to_numpy(float)[o:o + 24] for c in KNOWN_COV}} for o in pos]
    qs, _ = pipe.predict_quantiles(items, prediction_length=24, quantile_levels=pu.QUANTILES, batch_size=64)
    idx = m.index[np.concatenate([np.arange(o, o + 24) for o in pos])]
    return _quantile_frame(qs, idx)


# --- 주별 학습(매주 그 이전 7/1~ 자료만) ---------------------------------------------------------------
def weekly_oof(X, truth, track, seeds):
    """시험 주마다 피크 분류기와 LightGBM 회귀(부록 STEP 3 설정)를 학습해 그 주를 예측한다."""
    lab = truth['peak'].reindex(X.index).fillna(False).astype(bool)
    valid = truth['valid'].reindex(X.index).fillna(False).astype(bool)
    y = truth['y'].reindex(X.index)
    params = tuned_lgbm_params(track)
    parts = []
    for w in WEEKS.itertuples():
        tr = X.index[(X.index >= START) & (X.index < w.start) & valid.to_numpy()]
        te = X.index[(X.index >= w.start) & (X.index <= w.end)]
        parts.append(pd.DataFrame({'week': w.week,
                                   'p_clf': train_classifier(X, lab, tr, te, seeds),
                                   'lgbm': train_regressor(X, y, tr, te, params, w.start, seeds)}))
    return pd.concat(parts)


# --- 앞 주로 맞추고 다음 주에 적용(확률 보정·문턱) ------------------------------------------------------
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def forward_alarms(S, lab, ok, prob_paths, stack_cols=None):
    """S: 시험 기간 시각별 점수 표(week 열 포함). prob_paths의 각 점수는 isotonic 보정 후 F1 최대 문턱으로 경보한다.
    stack_cols가 있으면 로지스틱 스태킹 → isotonic → 문턱. 보정·문턱은 평가 주 이전의 주(1주차~)로만 맞춘다."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    out = []
    order = list(WEEKS['week'])
    for k in EVAL_WEEKS:
        prior = S['week'].isin(order[:order.index(k)]).to_numpy() & ok
        cur = (S['week'] == k).to_numpy()
        res = pd.DataFrame(index=S.index[cur])
        # 정답이 없는 시간(계측 중단 등)은 ok=False라 맞추기·평가에 쓰지 않는다. 계산이 멈추지 않게 결측만 0으로 채운다.
        raw = {p: np.nan_to_num(S[p].to_numpy(float)) for p in prob_paths}
        if stack_cols:
            F = np.nan_to_num(np.column_stack([_logit(S[c]) if c.startswith('p_') else S[c].to_numpy(float)
                                               for c in stack_cols]))
            st = LogisticRegression(C=1.0).fit(F[prior], lab[prior])
            raw['스태킹'] = st.predict_proba(F)[:, 1]
        for name, r in raw.items():
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(r[prior], lab[prior])
            pc_prior, pc_cur = iso.predict(r[prior]), iso.predict(r[cur])
            t = pu.best_threshold(lab[prior], pc_prior, None)
            res[f'{name}|p'] = pc_cur
            res[f'{name}|alarm'] = pc_cur >= t
            res[f'{name}|thr'] = t
        out.append(res)
    return pd.concat(out)


# --- 외부 데이터: 공개 철강 공장 15분 전력(UCI, CC BY 4.0)으로 Chronos-2 LoRA 전이학습 ----------------------
STEEL_CSV = ROOT / 'data' / 'external' / 'steel' / 'Steel_industry_data.csv'
STEEL_OFF_KW = 30.0                                        # 하루 평균이 이보다 낮으면 비가동일(분포가 두 덩어리로 갈림)
STEEL_VAL_DAYS = 28                                        # 체크포인트 선택 = 철강 마지막 28일(하루 단위 창)
FT_CUTOFF = WEEKS['start'].iloc[0]                         # 8/9: 출제 데이터는 이 시각 이전만 미세조정에 쓴다
FT_SEEDS = [42, 2021, 2024]
FT_VARIANTS = {'V2': ('small', False), 'V3': ('small', True), 'V4': ('base', False)}   # (모델, 출제 데이터 7/1~8/8 포함)


def load_steel():
    """UCI 철강 공장 15분 전력 → 시간 최대(kW)와 출제 데이터와 같은 형식의 미래 공변량(KNOWN_COV).
    - Usage_kWh는 15분 사용량이라 ×4 하면 15분 평균 kW(최대수요전력과 같은 단위)다.
    - 시각은 구간 끝 표기이고 자정 칸에는 그날 날짜가 붙어 있어, 행 순서로 구간 시작 시각을 만든다(날짜·요일 일치 확인).
    - 하루 종류는 01-2의 정의를 따른다: 비가동 = 휴일, 가동 토요일 = 토요일, 그 밖의 가동일 = 평일가동."""
    d = pd.read_csv(STEEL_CSV, encoding='utf-8-sig')
    ts = pd.Timestamp('2018-01-01') + pd.to_timedelta(np.arange(len(d)) * 15, 'min')
    label = pd.to_datetime(d['date'], format='%d/%m/%Y %H:%M')
    assert len(d) == 365 * 96 and d['Usage_kWh'].notna().all()
    assert (label.dt.normalize() == ts.normalize()).all() and (d['Day_of_week'].to_numpy() == ts.day_name()).all()
    kw = pd.Series(d['Usage_kWh'].to_numpy() * 4, index=ts)
    h = kw.resample('h').max().to_frame('y')
    op = kw.resample('D').mean() >= STEEL_OFF_KW
    dtype = pd.Series(np.where(~op, '휴일', np.where(op.index.dayofweek == 5, '토요일', '평일가동')), index=op.index)
    day, hr = h.index.normalize(), h.index.hour.to_numpy()
    h['hour_sin'], h['hour_cos'] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    for c, v in [('dt_op', '평일가동'), ('dt_sat', '토요일'), ('dt_off', '휴일')]:
        h[c] = dtype.eq(v).reindex(day).to_numpy(float)
    h['prev_op'] = op.shift(1, fill_value=True).reindex(day).to_numpy(float)
    return h


def ft_items(y, cov, ends):
    """계열 하나 → 끝 위치마다 [공변량 형태, 단변량 형태] 입력 두 개(트랙 A는 단변량, B는 공변량으로 예측하므로).
    Chronos-2는 형태가 섞인 목록을 받지 않아, 형태별로 전처리(PreparedInput)한 뒤 이어 붙인다."""
    from chronos.chronos2.preprocess import from_list_of_dicts
    with_cov = [{'target': y[:e], 'past_covariates': {c: cov[c][:e] for c in KNOWN_COV},
                 'future_covariates': {c: None for c in KNOWN_COV}} for e in ends]
    target_only = [{'target': y[:e]} for e in ends]
    return from_list_of_dicts(with_cov, prediction_length=24) + from_list_of_dicts(target_only, prediction_length=24)


def ft_data(steel, m, with_ours):
    """학습·검증 입력. 학습 = 철강(마지막 28일 제외) [+ 출제 데이터 7/1~8/8], 검증 = 철강 마지막 28일.
    학습 입력은 항목마다 같은 확률로 뽑히므로, 출제 데이터를 넣을 때는 철강 항목을 길이 비례만큼 반복한다."""
    ys = steel['y'].to_numpy(float)
    cs = {c: steel[c].to_numpy(float) for c in KNOWN_COV}
    n_tr = len(ys) - STEEL_VAL_DAYS * 24
    train = ft_items(ys, cs, [n_tr])
    val = ft_items(ys, cs, range(n_tr + 24, len(ys) + 1, 24))
    info = {'steel_train_hours': n_tr, 'steel_val_windows': STEEL_VAL_DAYS, 'ours_hours': 0, 'steel_repeat': 1}
    if with_ours:
        mo = m.loc[START:FT_CUTOFF - pd.Timedelta(hours=1)]
        assert mo.index.max() < FT_CUTOFF
        yo = mo['y_obs'].to_numpy(float)
        co = {c: v.to_numpy(float) for c, v in known_cov(mo).items()}
        rep = max(1, round(n_tr / len(yo)))
        train = train * rep + ft_items(yo, co, [len(yo)])
        info.update(ours_hours=len(yo), ours_last=str(mo.index.max()), steel_repeat=rep)
    return train, val, info


def finetune_chronos(size, train, val, seed, steps=2000, ctx=1344, every=250, tag='ft'):
    """부록 STEP 5의 미세조정 설정(lr 1e-5, 배치 16, 문맥 1344, every step마다 검증 → 최저 검증 손실 체크포인트).
    finetune_mode='lora'로 호출하지만, peft가 없는 환경에서는 chronos-forecasting이 전체 미세조정으로 실행한다."""
    import shutil
    import time
    from run_chronos import EvalLogger
    out_dir = OUT / 'logs' / f'_{tag}_ckpt'
    cb = EvalLogger()
    t0 = time.time()
    ft = chronos_pipe(size).fit(train, prediction_length=24, validation_inputs=val, finetune_mode='lora',
                                learning_rate=1e-5, num_steps=steps, batch_size=16, context_length=ctx,
                                output_dir=out_dir, callbacks=[cb], remove_printer_callback=True,
                                eval_steps=every, save_steps=every, logging_steps=50, seed=seed, data_seed=seed,
                                disable_tqdm=True)
    sec = time.time() - t0
    shutil.rmtree(out_dir, ignore_errors=True)                       # 체크포인트는 결과에 필요 없으므로 지움
    return ft, pd.DataFrame(cb.rows), sec
