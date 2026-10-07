"""STEP 4 — neuralforecast 딥러닝 모델(NHITS·TFT·LSTM)을 트랙 A·B로 학습·평가한다.

- 모델: NHITS, TFT, LSTM(가이드북 RNN 개선판: 외생변수 추가, 학습 구간에서만 맞춘 정규화로 스케일 누수 제거)
- 입력: y(시간 피크, outage는 직전 값으로 채우고 available_mask=0으로 손실에서 제외)
  futr_exog(미래에도 아는 값): 시각·요일 sin/cos, 비가동 요일(주말·공휴일), 교대, 과금 시간, TOU 시간대·계절
            + S1 당일 생산계획 + S2 당일 가동 여부·비가동 후 첫 가동일
  hist_exog(과거에만 아는 값): 그 시간의 15분 최솟값·마지막 15분 값·기울기, 시간 평균 사용량
- 정규화: neuralforecast의 창 단위 정규화는 드문 0/1 피처(공휴일 등)를 1e6 배로 키우는 문제가 있어 끄고(identity),
  y와 연속형 외생변수를 fold 학습 구간의 평균·표준편차로 직접 표준화한다(검증·테스트 정보 사용 안 함).
- 학습: cross_validation(refit=False)로 fold마다 1회 학습. val_size = 학습 구간 마지막 2주(336시간)로 early stopping
  (val_check_steps마다 검증, patience 회 동안 개선이 없으면 멈춤), max_steps 상한. random_seed=SEED, deterministic=True.
  트랙 A는 h=1로 336회, 트랙 B는 h=24로 14회(매일 00시) 예측한다.
- 기록: best step(검증 손실 최소 step), 멈춘 step, 학습 곡선(train/valid trajectories), 학습 시간, 1일 예측 시간

실행(프로젝트 루트에서): 작업 = 모델:트랙:시나리오[:입력길이][@fold], 예) nhits:A:S1, nhits:B:S1:336, tft:A:S1@F4
    PYTHONHASHSEED=0 N_THREADS=1 python scripts/train_neural.py --jobs nhits:A:S1@F1 --no-aggregate
    TFT는 CPU에서 매우 느려(축소 설정에서도 step당 약 3초) fast 모드(최근 3 fold: F4~F6)로만 돌린다.
    PYTHONHASHSEED=0 python scripts/train_neural.py --aggregate --jobs-for-aggregate <작업 목록>
결과(모으기 후): cv_scores.csv(model = nhits·tft·lstm, 입력 336이면 nhits_L336 등), oof parquet,
    outputs/tables/neural_runs.csv(fold·seed별 best step·학습 시간), outputs/logs/neural_trajectories.csv(학습 곡선)
"""
import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import power_utils as pu  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

warnings.filterwarnings('ignore')
for _lg in ('pytorch_lightning', 'lightning', 'lightning.pytorch', 'lightning.fabric', 'neuralforecast'):
    logging.getLogger(_lg).setLevel(logging.ERROR)
from neuralforecast import NeuralForecast  # noqa: E402
from neuralforecast.models import LSTM, NHITS, TFT  # noqa: E402

RUN_DIR = pu.OUT / 'logs' / 'neural_runs'
TRACK_CV = {'A': dict(h=1, step_size=1), 'B': dict(h=24, step_size=24)}
FUTR_BASE = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'offday', 'shift', 'billing_hour', 'tou_load', 'tou_season']
HIST = ['q_min', 'q_last', 'q_slope', 'usage']
# 공통 학습 설정: 50 step마다 검증, 5회(250 step) 동안 개선이 없으면 멈춤
COMMON = dict(scaler_type='identity', val_check_steps=50, early_stop_patience_steps=5, learning_rate=1e-3,
              enable_progress_bar=False, logger=False, enable_model_summary=False, accelerator='cpu', deterministic=True)
MODELS = {
    'nhits': (NHITS, dict(max_steps=1000)),
    'lstm': (LSTM, dict(max_steps=1000)),
    # TFT는 CPU에서 매우 느려 은닉 크기와 step당 창 수를 줄이고 step 상한을 낮춘다
    'tft': (TFT, dict(max_steps=500, hidden_size=32, windows_batch_size=256)),
}


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def build_frame(df, scenario, train_index):
    """neuralforecast 입력 표를 만든다. 표준화 기준(평균·표준편차)은 train_index(fold 학습 구간)에서만 구한다."""
    ok = ~df['outage']
    tr = df.index.isin(train_index) & ok
    y_in = pu.input_series(df)
    y_mu, y_sd = y_in[tr].mean(), y_in[tr].std()
    q = df[pu.QUARTER_COLS].where(ok).ffill()
    hist = pd.DataFrame({'q_min': q.min(axis=1), 'q_last': q['60분'], 'usage': df['usage'].where(ok).ffill(),
                         'q_slope': q.to_numpy() @ np.array([-1.5, -0.5, 0.5, 1.5]) / 5.0}, index=df.index)
    hist[['q_min', 'q_last', 'usage']] = (hist[['q_min', 'q_last', 'usage']] - y_mu) / y_sd
    hist['q_slope'] = hist['q_slope'] / y_sd
    h, dw = df['hour'].to_numpy(), df['dow'].to_numpy()
    futr = pd.DataFrame({'hour_sin': np.sin(2 * np.pi * h / 24), 'hour_cos': np.cos(2 * np.pi * h / 24),
                         'dow_sin': np.sin(2 * np.pi * dw / 7), 'dow_cos': np.cos(2 * np.pi * dw / 7),
                         'offday': (df['weekend'] | df['holiday']).astype(float),
                         'shift': df['shift'].astype(float), 'billing_hour': df['billing_hour'].astype(float),
                         'tou_load': df['tou_load_code'] / 2.0, 'tou_season': df['tou_season_code'] / 2.0},
                        index=df.index)
    futr_cols = list(FUTR_BASE)
    if scenario in ('S1', 'S2'):
        p = pu.fill_prod_plan(df[['prod_day']], train_index)['prod_day']
        futr['prod_day'] = (p - p[tr].mean()) / p[tr].std()
        futr_cols.append('prod_day')
    if scenario == 'S2':
        futr['op_today'] = df['op_day'].astype(float)
        futr['after_shutdown'] = df['after_shutdown'].astype(float)
        futr_cols += ['op_today', 'after_shutdown']
    frame = pd.DataFrame({'unique_id': 'y', 'ds': df.index, 'y': (y_in - y_mu) / y_sd,
                          'available_mask': ok.astype(float)}, index=df.index)
    frame = pd.concat([frame, futr[futr_cols], hist[HIST]], axis=1).reset_index(drop=True)
    assert not frame.drop(columns=['unique_id', 'ds']).isna().any().any(), '입력에 결측이 있습니다'
    return frame, futr_cols, (y_mu, y_sd)


def run_one(df, kind, track, scenario, input_size, fold, seed, max_steps=None):
    """fold 하나·seed 하나를 학습하고 검증 예측, 점수, 학습 곡선을 돌려준다."""
    pu.set_seed(seed)
    assert fold.val_end < pu.TEST_START, '테스트 구간이 섞였습니다'
    train_index = df.index[df.index < fold.val_start]
    frame, futr_cols, (mu, sd) = build_frame(df, scenario, train_index)
    upto = (df.index <= fold.val_end)
    frame = frame[upto].reset_index(drop=True)
    cls, cfg = MODELS[kind]
    if max_steps:                      # --smoke 시험용
        cfg = {**cfg, 'max_steps': max_steps}
    h, step = TRACK_CV[track]['h'], TRACK_CV[track]['step_size']
    model = cls(h=h, input_size=input_size, futr_exog_list=futr_cols, hist_exog_list=HIST, random_seed=seed,
                alias=kind, **COMMON, **cfg)
    nf = NeuralForecast(models=[model], freq='h')
    n_val_hours = int(((df.index >= fold.val_start) & upto).sum())
    t0 = time.time()
    cv = nf.cross_validation(df=frame, n_windows=n_val_hours // h, step_size=step, val_size=pu.ES_DAYS * 24, refit=False)
    fit_sec = time.time() - t0
    cv = cv.set_index('ds')
    assert cv.index.min() == fold.val_start and cv.index.max() == fold.val_end
    pred = pd.DataFrame({'y_pred': cv[kind].to_numpy() * sd + mu}, index=pd.DatetimeIndex(cv.index))
    # 1일 예측 시간: 학습된 모델로 마지막 시점 한 번 예측한 시간 × (24 / h)
    infer = np.nan
    try:
        last = frame['ds'].iloc[-(h + 1)]
        t1 = time.time()
        nf.predict(df=frame[frame['ds'] <= last], futr_df=frame[frame['ds'] > last][['unique_id', 'ds'] + futr_cols].head(h))
        infer = (time.time() - t1) * (24 // h)
    except Exception as e:  # 시간 측정 실패는 결과에 영향 없음
        log(f'추론 시간 측정 생략: {type(e).__name__}')
    m = nf.models[0]
    tr_traj = pd.DataFrame(m.train_trajectories, columns=['step', 'loss']).assign(kind='train')
    va_traj = pd.DataFrame(m.valid_trajectories, columns=['step', 'loss']).assign(kind='valid')
    best_step = int(va_traj.loc[va_traj['loss'].idxmin(), 'step']) if len(va_traj) else np.nan
    stopped = int(tr_traj['step'].max()) + 1 if len(tr_traj) else np.nan
    name = kind if input_size == 168 else f'{kind}_L{input_size}'
    sc = pu.evaluate(pred, df, train_index).assign(
        model=name, track=track, scenario=scenario, fold=fold.fold, seed=seed, deterministic=False,
        train_sec=fit_sec, infer_sec=infer)
    run = {'model': name, 'track': track, 'scenario': scenario, 'fold': fold.fold, 'seed': seed,
           'input_size': input_size, 'best_step': best_step, 'stopped_step': stopped,
           'max_steps': cfg['max_steps'], 'fit_sec': round(fit_sec, 1), 'infer_sec_per_day': infer}
    traj = pd.concat([tr_traj, va_traj]).assign(model=name, track=track, scenario=scenario, fold=fold.fold, seed=seed)
    return {'scores': sc, 'preds': pred.assign(fold=fold.fold, seed=seed), 'run': run, 'traj': traj}


def run_job(job, df, mode, n_seeds, smoke=False):
    job, _, only_fold = job.partition('@')            # 'nhits:A:S1@F3'처럼 fold 하나만 돌릴 수 있음(병렬용)
    parts = job.split(':')
    kind, track, scenario = parts[:3]
    input_size = int(parts[3]) if len(parts) > 3 else 168
    folds = pu.make_folds(mode).iloc[-1:] if smoke else pu.make_folds(mode)
    if only_fold:
        folds = folds[folds['fold'] == only_fold]
    run_dir = pu.OUT / 'logs' / 'neural_smoke' if smoke else RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for fold in folds.itertuples():
        for seed in pu.SEEDS[:1 if smoke else n_seeds]:
            path = run_dir / f"{job.replace(':', '_')}_{fold.fold}_s{seed}.pkl"
            if path.exists() and not smoke:
                out.append(pd.read_pickle(path))
                continue
            r = run_one(df, kind, track, scenario, input_size, fold, seed, max_steps=60 if smoke else None)
            pd.to_pickle(r, path)
            mae = r['scores'].loc[r['scores']['subset'] == 'all', 'mae'].iloc[0]
            log(f"{job} {fold.fold} seed {seed}: MAE {mae:.2f}, best step {r['run']['best_step']} / 멈춤 "
                f"{r['run']['stopped_step']}, {r['run']['fit_sec']:.0f}초")
            out.append(r)
    return out


def aggregate(df, jobs):
    results = []
    for job in jobs:
        paths = sorted(RUN_DIR.glob(f"{job.replace(':', '_')}_F*_s*.pkl"))
        assert paths, f'결과 없음: {job}'
        results += [pd.read_pickle(p) for p in paths]
    pu.upsert_cv_scores(pd.concat([r['scores'] for r in results], ignore_index=True))
    by_key = {}
    for r in results:
        k = (r['run']['model'], r['run']['track'], r['run']['scenario'])
        by_key.setdefault(k, []).append(r['preds'])
    for (name, track, scenario), preds in by_key.items():
        pu.save_oof(pd.concat(preds), df, name, track, scenario)
    pd.DataFrame([r['run'] for r in results]).to_csv(pu.OUT / 'tables' / 'neural_runs.csv', index=False,
                                                     encoding='utf-8-sig')
    pd.concat([r['traj'] for r in results]).to_csv(pu.OUT / 'logs' / 'neural_trajectories.csv', index=False,
                                                   encoding='utf-8-sig')
    log(f'모으기 완료: 학습 {len(results)}회, 모델 변형 {len(by_key)}개')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fast', 'full'], default='full')
    ap.add_argument('--jobs', nargs='*', default=[])
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--no-aggregate', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='마지막 fold·seed 1개·60 step 시험(outputs/logs/neural_smoke/)')
    args = ap.parse_args()
    torch.set_num_threads(pu.N_THREADS)
    df, _ = pu.load_data()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if not args.aggregate:
        for job in args.jobs:
            log(f'{job}: 시작 (mode={args.mode}, seed {args.n_seeds}개, 스레드 {pu.N_THREADS})')
            t0 = time.time()
            run_job(job, df, args.mode, args.n_seeds, smoke=args.smoke)
            log(f'{job}: 완료 {(time.time() - t0) / 60:.1f}분')
    if not args.no_aggregate and not args.smoke:
        aggregate(df, args.jobs)


if __name__ == '__main__':
    main()
