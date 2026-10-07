"""재현성·실행 환경 기록과 표·그림 저장(seed 고정, 버전·하드웨어 정보, 한글 글꼴, outputs/ 저장)."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from .config import N_THREADS, OUT


def set_seed(seed: int) -> None:
    """random·numpy·torch의 seed와 결정적 연산 설정을 한 번에 고정한다.

    PYTHONHASHSEED는 이미 실행 중인 파이썬에는 적용되지 않으므로,
    scripts/와 run_all.py는 실행 전에 환경변수로도 넣어 준다.
    GPU가 없으면 cudnn·CUBLAS 설정은 아무 효과가 없지만 다른 PC에서 같은 코드가 돌도록 둔다.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(N_THREADS)


def _cpu_name() -> str:
    """CPU 제품명을 돌려준다(Windows는 레지스트리, 그 외는 platform 정보)."""
    if platform.system() == 'Windows':
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _gpu_names() -> list[str]:
    """그래픽 카드 이름 목록(Windows만 조회, 실패하면 빈 목록)."""
    if platform.system() != 'Windows':
        return []
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             '(Get-CimInstance Win32_VideoController).Name'],
            capture_output=True, text=True, timeout=20)
        return [s.strip() for s in out.stdout.splitlines() if s.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


ENV_PACKAGES = ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'catboost', 'statsforecast',
                'neuralforecast', 'optuna', 'shap', 'pyarrow', 'matplotlib', 'seaborn', 'torch', 'keras',
                'chronos-forecasting', 'papermill', 'nbconvert']


def get_env_info() -> dict:
    """파이썬·패키지 버전과 하드웨어 정보를 사전(dict)으로 모은다. outputs/env_info.json에 저장한다."""
    versions = {}
    for pkg in ENV_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    info = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'cpu': _cpu_name(),
        'cpu_logical_cores': os.cpu_count(),
        'n_threads': N_THREADS,
        'gpu': _gpu_names(),
        'packages': versions,
    }
    try:
        import psutil
        info['ram_gb'] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        info['ram_gb'] = None
    try:
        import torch
        info['torch_cuda'] = torch.cuda.is_available()
        info['torch_device'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    except ImportError:
        info['torch_cuda'] = None
    return info


def save_env_info(info: dict, extra: dict | None = None) -> Path:
    """env_info를 outputs/env_info.json에 저장한다. extra(실행 시간 등)가 있으면 합친다."""
    path = OUT / 'env_info.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {**info, **(extra or {})}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def set_korean_font() -> str:
    """운영체제에 맞는 한글 글꼴을 matplotlib에 설정하고 글꼴 이름을 돌려준다."""
    import matplotlib as mpl
    from matplotlib import font_manager
    candidates = {'Windows': ['Malgun Gothic'], 'Darwin': ['AppleGothic', 'Apple SD Gothic Neo']}
    names = candidates.get(platform.system(), []) + ['NanumGothic', 'Noto Sans CJK KR']
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in names if n in available), 'DejaVu Sans')
    mpl.rcParams['font.family'] = chosen
    mpl.rcParams['axes.unicode_minus'] = False
    return chosen


def ensure_dirs() -> None:
    """outputs/ 아래 표준 폴더를 만든다."""
    for sub in ('tables', 'figures', 'preds', 'models', 'logs'):
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    """표를 outputs/tables/{name}.csv로 저장한다(엑셀에서 한글이 깨지지 않게 utf-8-sig)."""
    path = OUT / 'tables' / f'{name}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding='utf-8-sig')
    return path


def save_fig(fig, name: str) -> Path:
    """그림을 outputs/figures/{name}.png로 저장한다."""
    path = OUT / 'figures' / f'{name}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    return path


def run_script(script: str, args=(), log: str | None = None, threads: int | None = None, tail: int = 3) -> float:
    """scripts/의 학습·추론 스크립트를 별도 프로세스로 실행하고 걸린 시간(초)을 돌려준다(노트북 01에서 사용).

    seed 재현을 위해 PYTHONHASHSEED=0으로 고정하고, threads를 주면 N_THREADS 환경변수로 넘긴다.
    출력은 outputs/logs/{log}에 남기고 마지막 tail줄만 화면에 보여 준다. 실패하면 RuntimeError로 멈춘다.
    """
    from .config import ROOT
    env = {**os.environ, 'PYTHONHASHSEED': '0', 'PYTHONIOENCODING': 'utf-8'}
    if threads:
        env['N_THREADS'] = str(threads)
    log_path = OUT / 'logs' / (log or f'{Path(script).stem}.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as f:
        r = subprocess.run([sys.executable, '-W', 'ignore', str(ROOT / 'scripts' / script), *map(str, args)],
                           cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    sec = time.time() - t0
    last = [l for l in log_path.read_text(encoding='utf-8', errors='replace').splitlines() if l.strip()][-tail:]
    print(f'{script} {" ".join(map(str, args))[:80]} → {sec / 60:.1f}분 | ' + ' / '.join(last)[:300])
    if r.returncode != 0:
        raise RuntimeError(f'{script} 실패(로그: {log_path})')
    return sec
