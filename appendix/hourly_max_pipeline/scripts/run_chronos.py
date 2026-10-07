"""STEP 5 — 시계열 파운데이션 모델 Chronos-2를 zero-shot → 공변량 → LoRA 미세조정 순서로 평가한다.

API(chronos-forecasting 2.3.2, Apache-2.0): Chronos2Pipeline.from_pretrained(이름, device_map='cpu')
    .predict_quantiles(입력 리스트, prediction_length, quantile_levels)  입력 = 1차원 배열(단변량) 또는
        {'target': 배열, 'past_covariates': {이름: 배열}, 'future_covariates': {이름: 배열}} 딕셔너리
    .fit(inputs, prediction_length, validation_inputs, finetune_mode='lora', learning_rate, num_steps, ...)
모델: amazon/chronos-2(1억 2천만 파라미터), autogluon/chronos-2-small(2천8백만, CPU 미세조정용)

작업 종류(작업 이름 = 종류:모델:...@fold)
- zs:base:{A|B}:{문맥}@F1   단변량 zero-shot. 문맥 = 336·672·1344·2688·full(origin 이전 전체)
- cov:{base|small}:{A|B}:{S0|S1|S2}:{문맥}@F1   공변량 zero-shot
    미래에도 아는 값: 시각 sin·cos, 비가동일(주말·공휴일), 과금 시간, TOU 시간대 + S1 생산계획 + S2 가동 여부·비가동 후 첫 가동일
    과거에만 아는 값: 시간 평균 사용량
- lora:small:{S1}:{문맥}@F4:2021   LoRA 미세조정(학습 = 검증 시작 2주 전까지, 검증 = 학습 구간 마지막 2주의 하루 단위 14개)
    CPU 축소판: num_steps 1000까지 250 step마다 검증 손실을 재고 가장 낮은 체크포인트를 쓴다(250~1000 중 검증으로 선택),
    배치 16. CPU에서 배치 32·2000 step은 1회 5시간 이상이라 줄였다(계획: 500~2000). full 미세조정은 GPU(Colab)가 필요해 생략.
    24시간 앞으로 미세조정한 모델 하나로 트랙 A(1시간 앞)·B(24시간 앞)를 모두 예측한다.
공통: 0값(outage) 구간은 target을 NaN으로 넣어 마스크한다. zero-shot은 결정적이라 seed 반복이 없다(seed=0).
테스트 구간(9/1~9/14)은 학습·검증·예측 어디에도 쓰지 않는다.

실행(프로젝트 루트에서): PYTHONHASHSEED=0 N_THREADS=2 python scripts/run_chronos.py --jobs zs:base:A:336@F1 --no-aggregate
결과: outputs/logs/chronos_runs/*.pkl → --aggregate 로 cv_scores.csv·oof parquet·outputs/tables/chronos_*.csv 작성
"""
import argparse
import logging
import shutil
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
logging.getLogger('transformers').setLevel(logging.ERROR)
from chronos import Chronos2Pipeline  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

RUN_DIR = pu.OUT / 'logs' / 'chronos_runs'
MODEL_ID = {'base': 'amazon/chronos-2', 'small': 'autogluon/chronos-2-small'}
CONTEXTS = ['336', '672', '1344', '2688', 'full']
KNOWN = ['hour_sin', 'hour_cos', 'offday', 'billing_hour', 'tou_load']
PAST_ONLY = ['usage']
LORA_STEPS = 1000          # CPU 축소판 상한(250 step마다 검증)
_PIPES = {}


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def pipeline(size):
    if size not in _PIPES:
        _PIPES[size] = Chronos2Pipeline.from_pretrained(MODEL_ID[size], device_map='cpu')
    return _PIPES[size]


def covariates(df, scenario, train_index):
    """공변량 표(시간 단위). 생산계획 결손은 학습 구간 중앙값으로 채운다(미래 공변량에는 NaN을 넣지 않음)."""
    h = df['hour'].to_numpy()
    cov = pd.DataFrame({'hour_sin': np.sin(2 * np.pi * h / 24), 'hour_cos': np.cos(2 * np.pi * h / 24),
                        'offday': (df['weekend'] | df['holiday']).astype(float),
                        'billing_hour': df['billing_hour'].astype(float), 'tou_load': df['tou_load_code'].astype(float),
                        'usage': df['usage'].where(~df['outage'])}, index=df.index)
    known = list(KNOWN)
    if scenario in ('S1', 'S2'):
        cov['prod_day'] = pu.fill_prod_plan(df[['prod_day']], train_index)['prod_day']
        known.append('prod_day')
    if scenario == 'S2':
        cov['op_today'] = df['op_day'].astype(float)
        cov['after_shutdown'] = df['after_shutdown'].astype(float)
        known += ['op_today', 'after_shutdown']
    return cov, known


def origins(df, fold, track):
    """예측 시점 위치(정수 인덱스). 트랙 A = 검증 구간 매시, 트랙 B = 검증 구간 매일 00시."""
    va = np.where((df.index >= fold.val_start) & (df.index <= fold.val_end))[0]
    assert df.index[va].max() < pu.TEST_START, '테스트 구간이 섞였습니다'
    return va if track == 'A' else va[df['hour'].to_numpy()[va] == 0]


def make_items(y, cov, known, starts, ctx, h):
    """origin마다 문맥(직전 ctx시간, 'full'이면 처음부터)과 공변량으로 입력 하나를 만든다."""
    items = []
    for o in starts:
        lo = 0 if ctx == 'full' else max(0, o - int(ctx))
        if cov is None:
            items.append(y[lo:o])
        else:
            items.append({'target': y[lo:o],
                          'past_covariates': {c: cov[c].to_numpy(float)[lo:o] for c in known + PAST_ONLY},
                          'future_covariates': {c: cov[c].to_numpy(float)[o:o + h] for c in known}})
    return items


def predict_frame(pipe, items, starts, h, df):
    """분위수 예측을 시각별 표(q10~q90, y_pred=q50)로 바꾼다. 트랙 B는 origin마다 24행."""
    t0 = time.time()
    qs, _ = pipe.predict_quantiles(items, prediction_length=h, quantile_levels=pu.QUANTILES, batch_size=256)
    sec = time.time() - t0
    q = np.concatenate([qi[0].numpy() for qi in qs])                     # (origin 수 × h, 9)
    idx = df.index[np.concatenate([np.arange(o, o + h) for o in starts])]
    pf = pd.DataFrame(np.sort(q, axis=1), index=idx, columns=pu.Q_COLS)
    pf['y_pred'] = pf['q50']
    return pf, sec


class EvalLogger(TrainerCallback):
    """미세조정 중 검증 손실 기록(best step 확인용)."""

    def __init__(self):
        self.rows = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            self.rows.append({'step': state.global_step, 'eval_loss': metrics['eval_loss']})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            self.rows.append({'step': state.global_step, 'train_loss': logs['loss']})


def finetune(df, fold, scenario, ctx, seed):
    """chronos-2-small LoRA 미세조정. 학습 = 검증 시작 2주 전까지, 검증 = 그 뒤 2주를 하루 단위 14개로."""
    y = df['y_obs'].to_numpy(float)
    train_index = df.index[df.index < fold.val_start]
    cov, known = covariates(df, scenario, train_index)
    es = int(np.searchsorted(df.index, fold.val_start - pd.Timedelta(days=pu.ES_DAYS)))
    vs = int(np.searchsorted(df.index, fold.val_start))
    allc = known + PAST_ONLY
    train_in = [{'target': y[:es], 'past_covariates': {c: cov[c].to_numpy(float)[:es] for c in allc},
                 'future_covariates': {c: None for c in known}}]
    val_in = [{'target': y[:e], 'past_covariates': {c: cov[c].to_numpy(float)[:e] for c in allc},
               'future_covariates': {c: None for c in known}} for e in range(es + 24, vs + 1, 24)]
    out_dir = pu.OUT / 'models' / 'chronos_ft' / f'{fold.fold}_s{seed}_{scenario}'
    cb = EvalLogger()
    t0 = time.time()
    ft = pipeline('small').fit(train_in, prediction_length=24, validation_inputs=val_in, finetune_mode='lora',
                               learning_rate=1e-5, num_steps=LORA_STEPS, batch_size=16, context_length=int(ctx),
                               output_dir=out_dir, callbacks=[cb], remove_printer_callback=True,
                               eval_steps=250, save_steps=250, logging_steps=50, seed=seed, data_seed=seed,
                               disable_tqdm=True)
    fit_sec = time.time() - t0
    shutil.rmtree(out_dir, ignore_errors=True)                       # 체크포인트는 결과에 필요 없으므로 지움
    hist = pd.DataFrame(cb.rows)
    ev = hist.dropna(subset=['eval_loss']) if 'eval_loss' in hist else pd.DataFrame()
    best_step = int(ev.loc[ev['eval_loss'].idxmin(), 'step']) if len(ev) else np.nan
    return ft, cov, known, fit_sec, best_step, hist


def run_unit(df, unit):
    job, _, rest = unit.partition('@')
    fold_name, _, seed = rest.partition(':')
    fold = next(f for f in pu.make_folds('full').itertuples() if f.fold == fold_name)
    parts = job.split(':')
    kind = parts[0]
    y = df['y_obs'].to_numpy(float)
    train_index = df.index[df.index < fold.val_start]
    out = {'unit': unit, 'kind': kind, 'fold': fold.fold, 'seed': int(seed or 0), 'tracks': {}}
    if kind == 'lora':
        _, size, scenario, ctx = parts
        ft, cov, known, fit_sec, best_step, hist = finetune(df, fold, scenario, ctx, int(seed))
        out.update(size=size, scenario=scenario, ctx=ctx, fit_sec=fit_sec, best_step=best_step, history=hist)
        for track, h in (('A', 1), ('B', 24)):
            st = origins(df, fold, track)
            pf, sec = predict_frame(ft, make_items(y, cov, known, st, ctx, h), st, h, df)
            out['tracks'][track] = {'pred': pf, 'sec': sec}
    else:
        if kind == 'zs':
            _, size, track, ctx = parts
            scenario, cov, known = 'S0', None, None
        else:
            _, size, track, scenario, ctx = parts
            cov, known = covariates(df, scenario, train_index)
        h = 1 if track == 'A' else 24
        st = origins(df, fold, track)
        pf, sec = predict_frame(pipeline(size), make_items(y, cov, known, st, ctx, h), st, h, df)
        out.update(size=size, scenario=scenario, ctx=ctx)
        out['tracks'][track] = {'pred': pf, 'sec': sec}
    for track, r in out['tracks'].items():
        n_days = len(r['pred']) / 24
        r['scores'] = pu.evaluate(r['pred'], df, train_index).assign(infer_sec=r['sec'] / n_days)
    return out


def model_name(r):
    return {'zs': 'chronos2_zs', 'cov': 'chronos2_cov' if r['size'] == 'base' else 'chronos2s_cov',
            'lora': 'chronos2s_lora'}[r['kind']]


def aggregate(df):
    results = [pd.read_pickle(p) for p in sorted(RUN_DIR.glob('*.pkl'))]
    rows = []
    for r in results:
        for track, t in r['tracks'].items():
            rows.append(t['scores'].assign(kind=r['kind'], size=r['size'], ctx=r['ctx'], scenario=r['scenario'],
                                           track=track, fold=r['fold'], seed=r['seed'], model=model_name(r),
                                           train_sec=r.get('fit_sec', np.nan), deterministic=r['kind'] != 'lora'))
    allsc = pd.concat(rows, ignore_index=True)
    allsc.to_csv(pu.OUT / 'tables' / 'chronos_all_scores.csv', index=False, encoding='utf-8-sig')
    # 문맥 길이: 단변량 zero-shot에서 트랙별 검증 평균 MAE가 가장 작은 값을 고른다
    zs = allsc[(allsc['kind'] == 'zs') & (allsc['subset'] == 'all')]
    best_ctx = zs.groupby(['track', 'ctx'])['mae'].mean().groupby('track').idxmin().map(lambda k: k[1]).to_dict()
    keep = (allsc['kind'] != 'zs') | allsc.apply(lambda r: r['ctx'] == best_ctx.get(r['track']), axis=1)
    pu.upsert_cv_scores(allsc[keep].drop(columns=['kind', 'size', 'ctx']))
    preds = {}
    for r in results:
        for track, t in r['tracks'].items():
            if r['kind'] == 'zs' and r['ctx'] != best_ctx.get(track):
                continue                                  # 문맥 길이 실험의 나머지 길이는 표로만 남김
            preds.setdefault((model_name(r), track, r['scenario']), []).append(
                t['pred'].assign(fold=r['fold'], seed=r['seed']))
    for key, frames in preds.items():
        pu.save_oof(pd.concat(frames), df, *key)
    lora = [r for r in results if r['kind'] == 'lora']
    if lora:
        pd.DataFrame([{'fold': r['fold'], 'seed': r['seed'], 'scenario': r['scenario'], 'ctx': r['ctx'],
                       'best_step': r['best_step'], 'fit_sec': round(r['fit_sec'])} for r in lora]).to_csv(
            pu.OUT / 'tables' / 'chronos_lora_runs.csv', index=False, encoding='utf-8-sig')
        pd.concat([r['history'].assign(fold=r['fold'], seed=r['seed']) for r in lora]).to_csv(
            pu.OUT / 'logs' / 'chronos_lora_history.csv', index=False, encoding='utf-8-sig')
    log(f'모으기 완료: 단위 {len(results)}개, 선택된 문맥 길이 {best_ctx}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', nargs='*', default=[])
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--no-aggregate', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(pu.N_THREADS)
    df, _ = pu.load_data()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if not args.aggregate:
        for unit in args.jobs:
            path = RUN_DIR / (unit.replace(':', '_').replace('@', '__') + '.pkl')
            if path.exists():
                log(f'{unit}: 저장된 결과 사용')
                continue
            pu.set_seed(int(unit.partition('@')[2].partition(':')[2] or 0))
            t0 = time.time()
            r = run_unit(df, unit)
            pd.to_pickle(r, path)
            maes = {tr: t['scores'].loc[t['scores']['subset'] == 'all', 'mae'].iloc[0] for tr, t in r['tracks'].items()}
            log(f"{unit}: MAE {', '.join(f'{k} {v:.2f}' for k, v in maes.items())}, {time.time() - t0:.0f}초")
    if not args.no_aggregate:
        aggregate(df)


if __name__ == '__main__':
    main()
