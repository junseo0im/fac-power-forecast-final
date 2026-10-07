"""STEP 7 — 테스트 전에 정한 선정 규칙(노트북 02 [단계 ⑩])을 검증 결과에만 적용해 트랙별 최종 모델과 피크 경보 경로를 고른다.

테스트 예측(predict_test.py)보다 먼저 실행해 결과를 남긴다. 테스트 구간은 쓰지 않는다.
실행(프로젝트 루트에서): python scripts/step7_select.py
결과: outputs/tables/selection_table.csv(후보·검정·조건), outputs/tables/selection_result.json(트랙별 선정)
"""
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import power_utils as pu  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings('ignore')


def main():
    cv = pd.read_csv(pu.CV_PATH, encoding='utf-8-sig')
    cls = pd.read_csv(pu.OUT / 'tables' / 'step6_classification.csv', encoding='utf-8-sig')
    tables, result = [], {'선정 시각': time.strftime('%Y-%m-%d %H:%M:%S'), '테스트 사용': '아니오(검증 F2~F6만)'}
    for track in ['A', 'B']:
        t, pick = pu.apply_selection_rule(cv, track)
        f1, alarm = pu.select_alarm_path(cls, track)
        tables.append(t.assign(track=track))
        result[track] = pick
        result[f'{track}_alarm'] = alarm
        result[f'{track}_alarm_f1'] = {k: round(v, 4) for k, v in f1.items()}
        print(f'트랙 {track}: 점예측 {pick} | 피크 경보 {alarm}', flush=True)
    pd.concat(tables, ignore_index=True).to_csv(pu.OUT / 'tables' / 'selection_table.csv', index=False,
                                                encoding='utf-8-sig')
    (pu.OUT / 'tables' / 'selection_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                             encoding='utf-8')


if __name__ == '__main__':
    main()
