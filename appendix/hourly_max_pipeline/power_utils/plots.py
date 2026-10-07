"""예측 문제 정의 그림(트랙 A/B 예측 시점, 확장 윈도우 fold 타임라인)."""
from __future__ import annotations

import pandas as pd


def plot_track_timing():
    """트랙 A/B의 예측 시점·사용 가능 정보·예측 대상을 한 그림으로 보여준다."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True)
    cases = [('트랙 A (실시간 경보)', 34, 34, 35, '매 정시(예: 10시)에 다음 1시간 예측'),
             ('트랙 B (익일 계획)', 24, 24, 48, '매일 00시에 그날 24시간 예측')]
    for ax, (title, origin, t0, t1, note) in zip(axes, cases):
        ax.axvspan(0, origin, color='#9ecae1', alpha=0.7, label='사용 가능한 과거 정보')
        ax.axvspan(t0, t1, color='#fdae6b', alpha=0.9, label='예측 대상')
        ax.axvline(origin, color='k', lw=2)
        ax.text(origin, 1.02, ' 예측 시점', transform=ax.get_xaxis_transform(), fontsize=9)
        ax.set_yticks([]); ax.set_title(f'{title}: {note}', fontsize=10, loc='left')
        ax.legend(loc='upper left', fontsize=8, ncol=2)
    axes[-1].set_xticks(range(0, 49, 6))
    axes[-1].set_xticklabels(['D-1 00시', '06시', '12시', '18시', 'D 00시', '06시', '12시', '18시', 'D+1 00시'])
    axes[-1].set_xlim(0, 48)
    fig.tight_layout()
    return fig


def plot_fold_timeline(df: pd.DataFrame, folds: pd.DataFrame):
    """확장 윈도우 fold의 학습·검증 구간과 테스트 구간, 특이 구간(휴가·정전·결손)을 그린다."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(11, 3.8))
    start = df.index.min()
    for i, f in enumerate(folds.itertuples()):
        is_test = f.fold == 'TEST'
        ax.barh(i, f.val_start - start, left=start, color='#c6dbef', edgecolor='none')
        ax.barh(i, f.val_end - f.val_start, left=f.val_start, color='#e6550d' if is_test else '#3182bd')
    events = [('하계휴가', '2021-07-31', '2021-08-09'), ('정전', '2021-08-28 17:00', '2021-08-29 12:00'),
              ('7/13·15 결손', '2021-07-13', '2021-07-16')]
    for label, a, b in events:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='grey', alpha=0.25)
        ax.text(pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2, -0.75, label,
                fontsize=8, ha='center', va='center')
    ax.set_yticks(range(len(folds))); ax.set_yticklabels(folds['fold'])
    ax.set_ylim(len(folds) - 0.5, -1.1)           # 위쪽에 특이 구간 이름을 쓸 여백
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_xlim(start, df.index.max())
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#c6dbef', label='학습(확장 윈도우)'), Patch(color='#3182bd', label='검증 2주'),
                       Patch(color='#e6550d', label='테스트(마지막 1회)'), Patch(color='grey', alpha=0.25, label='특이 구간')],
              loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=8, ncol=4, frameon=False)
    fig.tight_layout()
    return fig
