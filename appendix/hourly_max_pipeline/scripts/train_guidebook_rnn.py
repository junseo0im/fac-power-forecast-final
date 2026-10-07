"""STEP 2 — 가이드북 RNN을 같은 타깃(시간 피크 y)·같은 시간 fold로 다시 학습·평가한다.

가이드북과 같은 것: SimpleRNN 50 → Flatten → Dense 1, 168시차를 (24, 7)로 자른 입력, Adam(lr=1e-4), MSE, batch 32
가이드북과 다른 것:
- 타깃: '15분' 컬럼 → 시간 피크 y = max(15·30·45·60분)
- 스케일러: 전체 구간 fit → fold 학습 구간에서만 fit
- epoch: 500 고정 → 학습 구간 마지막 2주를 검증으로 쓴 early stopping(patience 30, 최대 500, 최적 가중치 복원)
- 평가: 9/1~9/14 1회 → 확장 윈도우 검증 6 fold × seed 3개. 트랙 A(1시간 앞), 트랙 B(1시간 앞 모델로 24회 재귀 예측)

실행(프로젝트 루트에서, 오래 걸리므로 백그라운드 권장):
    PYTHONHASHSEED=0 python scripts/train_guidebook_rnn.py --mode full > outputs/logs/guidebook_rnn.log 2>&1
    --smoke : F6·seed 42·3 epoch만 돌려 속도만 확인(공식 결과는 쓰지 않음)
    --resume: 이미 끝난 fold·seed 결과(outputs/logs/guidebook_rnn_runs/)는 다시 돌리지 않음
    --folds F1 F2 --no-aggregate : 일부 fold만 학습하고 결과 모으기는 건너뜀(병렬 실행용)
병렬 실행: 작은 RNN은 스레드 1개가 가장 빠르므로(epoch당 약 2.2초, 6스레드는 3.6초) fold마다 프로세스를 따로 띄운다.
    for f in F1 F2 F3 F4 F5 F6; do N_THREADS=1 PYTHONHASHSEED=0 python scripts/train_guidebook_rnn.py \
        --folds $f --no-aggregate > outputs/logs/guidebook_rnn_$f.log 2>&1 & done; wait
    PYTHONHASHSEED=0 python scripts/train_guidebook_rnn.py --resume   # 저장된 18개 결과를 모아 공식 파일 작성
결과
- outputs/tables/cv_scores.csv              : model='guidebook_rnn', 트랙 A·B
- outputs/preds/oof_guidebook_rnn_{A,B}_S0.parquet
- outputs/tables/guidebook_rnn_epochs.csv    : fold·seed별 best epoch, 멈춘 epoch, 학습 시간
- outputs/logs/guidebook_rnn_history.csv     : fold·seed·epoch별 학습/검증 손실(학습 곡선용)
- outputs/logs/guidebook_rnn_summary.json    : 테스트 재학습 epoch(best epoch 중앙값), 총 실행 시간
테스트 구간(9/1~9/14)은 학습·예측·평가 어디에도 쓰지 않는다. 테스트 재학습은 STEP 7에서 위 epoch로 한다.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ['KERAS_BACKEND'] = 'torch'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import power_utils as pu  # noqa: E402  (스레드 수 고정을 위해 numpy보다 먼저)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import keras  # noqa: E402

MODEL = 'guidebook_rnn'
RUN_DIR = pu.OUT / 'logs' / 'guidebook_rnn_runs'


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def run_one(df, lags, y_hist, fold, seed, max_epochs, patience):
    """fold 하나·seed 하나를 학습하고 트랙 A·B 예측과 점수를 돌려준다."""
    pu.set_seed(seed)
    keras.utils.set_random_seed(seed)
    idx = df.index
    target = df['y_obs'].to_numpy()
    usable = (np.arange(len(idx)) >= pu.RNN_LAGS) & ~np.isnan(target)    # outage 타깃은 학습에서 뺌
    es_start = fold.val_start - pd.Timedelta(days=pu.ES_DAYS)
    tr = usable & (idx < es_start)
    es = usable & (idx >= es_start) & (idx < fold.val_start)
    va = (idx >= fold.val_start) & (idx <= fold.val_end)
    assert idx[va].max() < pu.TEST_START, '테스트 구간이 섞였습니다'

    sc = pu.GuidebookScaler().fit(target[tr], lags[tr])
    to_x = lambda m: sc.transform_lags(m).reshape(-1, 24, 7)
    model = pu.build_guidebook_rnn()
    stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    t0 = time.time()
    hist = model.fit(to_x(lags[tr]), sc.transform_y(target[tr]),
                     validation_data=(to_x(lags[es]), sc.transform_y(target[es])),
                     epochs=max_epochs, batch_size=32, shuffle=True, verbose=0, callbacks=[stop])
    train_sec = time.time() - t0
    best_epoch = int(np.argmin(hist.history['val_loss'])) + 1

    def predict(lag_rows):
        out = model.predict(to_x(lag_rows), batch_size=1024, verbose=0).ravel()
        return sc.inverse_y(out)

    n_days = int(va.sum() // 24)
    t0 = time.time()
    pred_a = predict(lags[va])                                   # 트랙 A: 실제 직전 168시간으로 1시간 앞
    infer_a = (time.time() - t0) / n_days
    origins = np.where(va & (df['hour'].to_numpy() == 0))[0]     # 트랙 B: 매일 00시에 24회 재귀
    history = np.stack([y_hist[o - pu.RNN_LAGS:o] for o in origins])
    t0 = time.time()
    pred_b = pu.recursive_forecast(predict, history, steps=24).ravel()
    infer_b = (time.time() - t0) / n_days

    train_index = idx[idx < fold.val_start]
    scores, preds = [], []
    for track, p, infer in (('A', pred_a, infer_a), ('B', pred_b, infer_b)):
        pf = pd.DataFrame({'y_pred': p}, index=idx[va])
        sc_df = pu.evaluate(pf, df, train_index)
        sc_df = sc_df.assign(model=MODEL, track=track, scenario='S0', fold=fold.fold, seed=seed,
                             deterministic=False, train_sec=train_sec, infer_sec=infer)
        scores.append(sc_df)
        preds.append(pf.assign(fold=fold.fold, seed=seed, track=track))
    h = pd.DataFrame(hist.history)
    h.insert(0, 'epoch', np.arange(1, len(h) + 1))
    epoch_row = {'fold': fold.fold, 'seed': seed, 'best_epoch': best_epoch, 'stopped_epoch': len(h),
                 'best_val_loss': float(np.min(hist.history['val_loss'])), 'train_sec': round(train_sec, 1),
                 'n_train': int(tr.sum()), 'n_es': int(es.sum())}
    return {'scores': pd.concat(scores), 'preds': pd.concat(preds),
            'history': h.assign(fold=fold.fold, seed=seed), 'epoch': epoch_row}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fast', 'full'], default='full')
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--max-epochs', type=int, default=500)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--folds', nargs='*', help='이 fold만 학습(예: F1 F2)')
    ap.add_argument('--no-aggregate', action='store_true', help='학습만 하고 공식 결과 파일은 쓰지 않음')
    args = ap.parse_args()

    df, _ = pu.load_data()
    y_hist = pu.input_series(df).to_numpy()
    lags = pu.lag_matrix(pu.input_series(df))
    folds = pu.make_folds(args.mode)
    if args.folds:
        folds = folds[folds['fold'].isin(args.folds)]
    seeds = pu.SEEDS[:args.n_seeds]
    if args.smoke:
        folds, seeds, args.max_epochs = folds.iloc[[-1]], seeds[:1], 3
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log(f'시작: mode={args.mode}, fold {list(folds.fold)}, seed {seeds}, 최대 {args.max_epochs} epoch, '
        f'patience {args.patience}, 스레드 {pu.N_THREADS}, keras 백엔드 {keras.backend.backend()}')

    t_all = time.time()
    results = []
    for fold in folds.itertuples():
        for seed in seeds:
            path = RUN_DIR / f'{fold.fold}_s{seed}.pkl'
            if args.resume and path.exists() and not args.smoke:
                results.append(pd.read_pickle(path))
                log(f'{fold.fold} seed {seed}: 저장된 결과 사용')
                continue
            r = run_one(df, lags, y_hist, fold, seed, args.max_epochs, args.patience)
            s = r['scores'].query("subset == 'all'").set_index('track')['mae']
            log(f"{fold.fold} seed {seed}: best epoch {r['epoch']['best_epoch']} / 멈춤 {r['epoch']['stopped_epoch']}, "
                f"학습 {r['epoch']['train_sec']:.0f}초, MAE A {s['A']:.2f} · B {s['B']:.2f}")
            if args.smoke:
                log(f"smoke 완료: epoch당 약 {r['epoch']['train_sec'] / r['epoch']['stopped_epoch']:.2f}초")
                return
            pd.to_pickle(r, path)
            results.append(r)

    if args.no_aggregate:
        log(f'학습 완료(모으기 생략): {len(results)}회, {(time.time() - t_all) / 60:.1f}분')
        return
    assert len(results) == len(pu.make_folds(args.mode)) * len(seeds), '모든 fold·seed 결과가 있어야 모을 수 있습니다'
    scores = pd.concat([r['scores'] for r in results], ignore_index=True)
    pu.upsert_cv_scores(scores)
    preds = pd.concat([r['preds'] for r in results])
    for track in ('A', 'B'):
        pu.save_oof(preds[preds['track'] == track].drop(columns='track'), df, MODEL, track, 'S0')
    epochs = pd.DataFrame([r['epoch'] for r in results])
    epochs.to_csv(pu.OUT / 'tables' / 'guidebook_rnn_epochs.csv', index=False, encoding='utf-8-sig')
    pd.concat([r['history'] for r in results]).to_csv(pu.OUT / 'logs' / 'guidebook_rnn_history.csv',
                                                      index=False, encoding='utf-8-sig')
    summary = {'mode': args.mode, 'seeds': seeds, 'folds': list(folds.fold),
               'test_refit_epochs': int(np.median(epochs['best_epoch'])),
               'train_sec_sum': round(float(epochs['train_sec'].sum()), 1),
               'note': '학습은 fold별 프로세스 6개를 스레드 1개씩 병렬 실행', 'env': pu.get_env_info()}
    (pu.OUT / 'logs' / 'guidebook_rnn_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f"완료: {len(results)}회 학습, 학습 시간 합계 {summary['train_sec_sum'] / 60:.1f}분, "
        f"테스트 재학습 epoch(중앙값) {summary['test_refit_epochs']}")


if __name__ == '__main__':
    main()
