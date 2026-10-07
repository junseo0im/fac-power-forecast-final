"""확장 윈도우 검증 fold와, 이전 fold로 맞추고 다음 fold에 적용하는 2단계 검증 순서."""
from __future__ import annotations

import pandas as pd

from .config import TEST_END, TEST_START, VAL_DAYS, VAL_STARTS


def make_folds(mode: str = 'full', include_test: bool = False) -> pd.DataFrame:
    """확장 윈도우 검증 fold 표를 만든다. 각 fold는 val_start 이전 데이터만으로 학습한다.

    mode='fast'면 최근 3 fold(F4~F6)만 쓴다. include_test=True면 마지막에 TEST(9/1~9/14)를 붙인다.
    TEST 행은 STEP 7의 최종 1회 평가에서만 쓴다.
    """
    rows = []
    for i, s in enumerate(VAL_STARTS, start=1):
        start = pd.Timestamp(s)
        rows.append({'fold': f'F{i}', 'val_start': start,
                     'val_end': start + pd.Timedelta(days=VAL_DAYS) - pd.Timedelta(hours=1)})
    folds = pd.DataFrame(rows)
    if mode == 'fast':
        folds = folds.iloc[-3:]
    if include_test:
        folds = pd.concat([folds, pd.DataFrame([{'fold': 'TEST', 'val_start': TEST_START, 'val_end': TEST_END}])])
    return folds.reset_index(drop=True)


def fold_index(index: pd.DatetimeIndex, fold) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """fold 한 행으로 (학습 인덱스, 평가 인덱스)를 돌려준다. 학습은 val_start 이전 전부(확장 윈도우)."""
    train = index[index < fold['val_start']]
    val = index[(index >= fold['val_start']) & (index <= fold['val_end'])]
    return train, val


def forward_folds(folds) -> list[tuple[str, list[str]]]:
    """시간 순서를 지키는 2단계 검증용 (평가 fold, 그 이전 fold들) 목록. 첫 fold는 이전 fold가 없어 뺀다.

    앙상블 가중치·컨포멀 보정·확률 보정·임계값은 '이전 fold들의 검증 예측'으로만 맞추고 평가 fold에 적용한다.
    테스트(STEP 7)에는 6개 fold 전체로 맞춘 값을 한 번만 적용한다.
    """
    names = list(folds)
    return [(names[i], names[:i]) for i in range(1, len(names))]
