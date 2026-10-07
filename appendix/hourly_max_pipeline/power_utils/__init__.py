"""전력 수요예측·최대피크 위험 프로젝트 공통 함수 패키지.

노트북(notebooks/02_AI예측모델개발.ipynb)과 scripts/*.py가 `import power_utils as pu`로 불러 쓴다.
특히 아래 두 가지 규칙을 코드로 강제한다.
- 피처는 예측 시점에 알 수 있는 값만 쓴다(features.make_features의 assert, features.check_no_leak).
- 테스트 구간(2021-09-01~09-14)은 튜닝·선정·보정에 쓰지 않는다(scripts/predict_test.py에서만 사용).

모듈 구성(위에서 아래로 의존):
- config     : 경로·상수·실행 설정. 스레드 수 환경변수는 numpy보다 먼저 정해야 하므로 이 모듈이 가장 먼저 불린다.
- env        : 재현성·실행 환경 기록과 표·그림 저장(seed 고정, 버전·하드웨어 정보, 한글 글꼴, outputs/ 저장).
- data       : 데이터 불러오기와 플래그: 시각 재구성, 시간 피크 y, 0값(outage)·가동일·공휴일·복사일·생산계획, 한전 TOU, 15분 long 시계열.
- features   : 트랙·시나리오별 피처 생성과 누수 검사(피처 가용성 표, 정적 assert, 동적 교란 검사).
- folds      : 확장 윈도우 검증 fold와, 이전 fold로 맞추고 다음 fold에 적용하는 2단계 검증 순서.
- evaluation : 평가 지표(회귀·확률·분류)와 결과 저장·불러오기(cv_scores.csv, 검증 예측 parquet, 요약표).
- baselines  : 나이브 기준선 3종(직전값, 1주 전 같은 시각, 최근 4주 같은 요일·시각 평균).
- guidebook  : 가이드북 베이스라인(RNN·RF) 재평가용 함수(원래 수치 읽기, 168시차, 열별 MinMax, RNN 구조, 재귀 예측, RF 피처).
- models     : 트리·통계 모델 공통 설정(학습 가중치, Optuna 탐색 공간, LightGBM·XGBoost·CatBoost 생성, ARIMA 푸리에·외생변수).
- ensemble   : 불확실성·확률 도구(분위수 → 피크 초과 확률, 컨포멀 분위수 회귀, 경보 임계값 선택).
- selection  : 유의성 검정(Diebold–Mariano HLN, 블록 부트스트랩)과 테스트 전에 정한 최종 선정 규칙.
- plots      : 예측 문제 정의 그림(트랙 A/B 예측 시점, 확장 윈도우 fold 타임라인).
"""

from .config import N_THREADS, ROOT, DATA_PATH, OUT, SEEDS, QUANTILES, Q_COLS, QUARTER_COLS, WEATHER_COLS, OP_THRESHOLD, VAL_STARTS, VAL_DAYS, TEST_START, TEST_END, HOLIDAYS_2021, BASELINES, CV_PATH, RNN_LAGS, ES_DAYS  # noqa: F401
from .env import set_seed, _cpu_name, _gpu_names, ENV_PACKAGES, get_env_info, save_env_info, set_korean_font, ensure_dirs, save_table, save_fig, run_script  # noqa: F401
from .data import tou_2021, load_data, _prod_bin, _to_quarter_long, input_series, lag_matrix, _DF_CACHE, load_data_cached  # noqa: F401
from .features import GROUP_CAL, GROUP_LAG, GROUP_S1, GROUP_S2, GROUP_WX, KIND_LABEL, make_features, assert_feature_availability, _avail_reason, check_no_leak, fill_prod_plan  # noqa: F401
from .folds import make_folds, fold_index, forward_folds  # noqa: F401
from .evaluation import _ece, _cls_metrics, evaluate, CV_COLS, upsert_cv_scores, save_oof, load_oof, mae_by, oof_seed_mean, summarize_scores  # noqa: F401
from .baselines import baseline_predict, run_baselines  # noqa: F401
from .guidebook import BASELINE_NB, guidebook_reference, GuidebookScaler, build_guidebook_rnn, recursive_forecast, RF_FEATURES, guidebook_rf_features  # noqa: F401
from .models import TAU_CHOICES, PEAK_ALPHAS, TREE_MODELS, sample_weight, suggest_tree_params, make_tree_model, fourier_terms, arima_exog  # noqa: F401
from .ensemble import peak_prob_from_quantiles, conformal_adjust, best_threshold  # noqa: F401
from .selection import dm_test, block_bootstrap_ci, SIMPLICITY, SELECT_FOLDS, ALARM_ORDER, apply_selection_rule, select_alarm_path  # noqa: F401
from .plots import plot_track_timing, plot_fold_timeline  # noqa: F401
