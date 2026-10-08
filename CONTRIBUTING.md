# 기여 방법

팀원이 같은 방식으로 작업 기록을 남기도록 정한 규칙입니다.

## 커밋 메시지

`종류: 한 줄 설명` 형식으로 씁니다. 설명은 한국어로, 무엇을 왜 바꿨는지가 드러나게 씁니다.

| 종류 | 쓰는 경우 | 예 |
|---|---|---|
| `feat` | 새 분석, 새 기능, 새 노트북 절 | `feat: 옮길 양을 전날 예측으로 정하는 배치 방안 추가` |
| `fix` | 계산·코드 오류 수정 | `fix: 부록 표 M24의 장치를 측정 기록에서 읽도록 수정` |
| `docs` | 보고서, README, 참고 자료, 발표자료 | `docs: 3장 경보 단위 설명 보완` |
| `refactor` | 결과가 바뀌지 않는 코드 정리 | `refactor: 요금 시간대 함수를 한 곳으로 모음` |
| `test` | 자동 테스트 추가·수정 | `test: 1시간 앞 예측 발행 시각 점검 추가` |
| `chore` | 패키지 버전, 실행 스크립트, 설정 파일 | `chore: SciPy 버전 고정` |

- 결과 숫자가 바뀌는 커밋은 본문에 무엇이 얼마나 바뀌었는지 적습니다. 예: `여름 최대 203.5kW 유지, 새로 1단계를 넘은 구간 16 → 13`
- 한 커밋에는 한 가지 목적만 담습니다. 분석 수정과 보고서 문장 수정은 나눕니다.

## 작업 순서

1. 노트북을 고친 뒤에는 `python run_all.py --check`로 다시 실행해 `outputs/`를 갱신하고, 바뀐 결과표가 의도한 것뿐인지 확인합니다.
2. `python -m pytest tests -q`가 모두 통과하는지 확인합니다.
3. 숫자가 바뀌었다면 [REPORT.md](REPORT.md)와 README, `docs/` 문서를 함께 맞춥니다.
4. 노트북 출력에 원자료 행이나 개인 경로가 찍히지 않았는지 확인합니다.

## 올리지 않는 것

- 원본 데이터(`data/raw/`)와 전처리 결과(`data/processed/`)
- 외부 자료(`data/external/`)와 미세조정 체크포인트(`outputs/chronos_ft/`)
- 실제 전력값이 들어 있는 예측·판단 기록: `outputs/predictions/`에서 `chronos_*`, `extra_*` 외 파일, `outputs/preds/`, 루트의 `테스트데이터_예측결과.csv`
- 부록의 실제값 파일: `appendix/hourly_max_pipeline/outputs/preds/`, `outputs/submission/`, 표 `step6_*`·`shap_features_*`
- 블라인드 검사용 단어 목록(`blind_words.txt`), 제출 zip
