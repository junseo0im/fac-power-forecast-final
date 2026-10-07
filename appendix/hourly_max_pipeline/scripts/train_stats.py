"""STEP 3 — 통계 모델(MSTL + AutoETS 추세, AutoARIMA 동적 조화 회귀)을 트랙 A·B로 평가한다. 결정적이라 seed 반복은 없다.

- mstl  : MSTL(season_length=[24, 168]) + AutoETS(추세) — 외생변수를 못 넣으므로 S0만
- arima : AutoARIMA(season_length=1) + 외생변수(일·주 푸리에 항, 공휴일 / S1 생산계획 / S2 가동 여부) — S0·S1·S2
- fold마다 검증 시작 이전 데이터로 한 번 학습하고, statsforecast cross_validation(refit=False)로
  트랙 A는 매시 1시간 앞(336회), 트랙 B는 매일 00시에 24시간 앞(14회)을 예측한다. 학습 후에는 실제값으로 상태만 갱신한다.
- 80% 예측구간(lo-80·hi-80)을 q10·q90으로 저장한다. 입력 y의 outage(NaN)는 직전 값으로 채우고, 평가에서는 뺀다.

실행(프로젝트 루트에서): PYTHONHASHSEED=0 python scripts/train_stats.py > outputs/logs/stats.log 2>&1
    --jobs mstl:S0 arima:S1 --no-aggregate : 일부만 실행(병렬용), --aggregate : 저장된 결과 모으기
결과: cv_scores.csv(model='mstl'·'arima'), outputs/preds/oof_{model}_{track}_{scenario}.parquet
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import power_utils as pu  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsforecast import StatsForecast  # noqa: E402
from statsforecast.models import MSTL, AutoARIMA, AutoETS  # noqa: E402

RUN_DIR = pu.OUT / 'logs' / 'stats_runs'
JOBS = ['mstl:S0', 'arima:S0', 'arima:S1', 'arima:S2']
TRACK_CV = {'A': dict(h=1, step_size=1), 'B': dict(h=24, step_size=24)}


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def make_model(name):
    if name == 'mstl':
        return MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model='ZZN'), alias=name)
    return AutoARIMA(season_length=1, alias=name)


def run_job(job, df, folds):
    name, scenario = job.split(':')
    y_in = pu.input_series(df)
    scores, preds = [], {}
    t_job = time.time()
    for f in folds.itertuples():
        assert f.val_end < pu.TEST_START, '테스트 구간이 섞였습니다'
        upto = df.index <= f.val_end
        sdf = pd.DataFrame({'unique_id': 'y', 'ds': df.index[upto], 'y': y_in[upto].to_numpy()})
        if name == 'arima':
            exog = pu.arima_exog(df, scenario, df.index[df.index < f.val_start]).loc[upto]
            sdf = pd.concat([sdf, exog.reset_index(drop=True)], axis=1)
        n_hours = int(((df.index >= f.val_start) & upto).sum())
        full_tr = df.index[df.index < f.val_start]
        for track, cvargs in TRACK_CV.items():
            t0 = time.time()
            sf = StatsForecast(models=[make_model(name)], freq='h', n_jobs=1)
            cv = sf.cross_validation(df=sdf, n_windows=n_hours // cvargs['h'], refit=False, level=[80], **cvargs)
            sec = time.time() - t0
            cv = cv.set_index('ds')
            assert cv.index.min() == f.val_start and cv.index.max() == f.val_end
            pf = pd.DataFrame({'y_pred': cv[name], 'q10': cv[f'{name}-lo-80'], 'q90': cv[f'{name}-hi-80']})
            sc = pu.evaluate(pf, df, full_tr)
            # 학습 1회 + 갱신·예측 시간을 함께 잰 값이라 train_sec에 넣고, 1일당 시간은 infer_sec로 나눠 기록
            scores.append(sc.assign(model=name, track=track, scenario=scenario, fold=f.fold, seed=0,
                                    deterministic=True, train_sec=sec, infer_sec=sec / (n_hours / 24)))
            preds.setdefault(track, []).append(pf.assign(fold=f.fold, seed=0))
            log(f"{job} {f.fold} 트랙 {track}: MAE {sc.loc[sc['subset'] == 'all', 'mae'].iloc[0]:.2f}, {sec:.0f}초")
    return {'job': job, 'scores': pd.concat(scores, ignore_index=True),
            'preds': {k: pd.concat(v) for k, v in preds.items()}, 'total_sec': time.time() - t_job}


def aggregate(df):
    paths = sorted(RUN_DIR.glob('*.pkl'))
    missing = {j.replace(':', '_') for j in JOBS} - {p.stem for p in paths}
    assert not missing, f'끝나지 않은 작업: {sorted(missing)}'
    results = [pd.read_pickle(p) for p in paths]
    pu.upsert_cv_scores(pd.concat([r['scores'] for r in results], ignore_index=True))
    for r in results:
        name, scenario = r['job'].split(':')
        for track, pf in r['preds'].items():
            pu.save_oof(pf, df, name, track, scenario)
    log(f'모으기 완료: 작업 {len(results)}개')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fast', 'full'], default='full')
    ap.add_argument('--jobs', nargs='*')
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--no-aggregate', action='store_true')
    args = ap.parse_args()
    df, _ = pu.load_data()
    folds = pu.make_folds(args.mode)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for job in (args.jobs or ([] if args.aggregate else JOBS)):
        log(f'{job}: 시작 (fold {list(folds.fold)})')
        res = run_job(job, df, folds)
        pd.to_pickle(res, RUN_DIR / f"{job.replace(':', '_')}.pkl")
        log(f"{job}: 완료 {res['total_sec'] / 60:.1f}분")
    if not args.no_aggregate:
        aggregate(df)


if __name__ == '__main__':
    main()
