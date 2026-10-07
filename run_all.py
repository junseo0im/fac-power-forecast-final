"""전체 실행: 데이터 진단 → 전처리 → 예측모델 → 피크 경보 → 영향요인·오류분석 → 현장 활용 → 테스트 예측결과 파일.

사용법(프로젝트 루트에서, requirements.txt를 설치한 파이썬으로)
    python run_all.py                  # 기본: 02-1·02-1b·02-1c(Chronos-2, 딥러닝·통계, 미세조정)는 저장된 예측을 쓰고 나머지 노트북을 처음부터 실행
    python run_all.py --with-chronos   # 02-1·02-1b·02-1c도 다시 실행(예측을 새로 계산, CPU로 수 시간)
    python run_all.py --appendix       # 부록 파이프라인 노트북(01→02→03)도 저장된 결과로 다시 실행
    python run_all.py --check          # 실행 전후로 outputs/tables의 표가 같은지 비교

- 노트북은 각 노트북 폴더를 작업 폴더로 하여 처음부터 끝까지 실행하고, 실행 결과를 노트북에 그대로 저장한다.
- 노트북은 이 스크립트를 실행한 파이썬으로 돌린다(다른 파이썬의 jupyter·커널 설정을 쓰지 않도록 커널 설정을 실행 때마다 만듦).
- --check는 표의 글자가 같은지 보고, 다르면 숫자를 상대오차 1e-9 안에서 다시 비교한다. CPU·파이썬 버전에 따라 생기는
  소수 열 자리 이하의 차이는 '부동소수점 차이만'으로 따로 센다.
- 실행 시간·환경 정보는 outputs/run_all_summary.json에 남긴다. 마지막에 tests/를 pytest로 실행한다(pytest가 있을 때).
"""
import argparse
import csv
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB = ROOT / 'notebooks'
APPX = ROOT / 'appendix' / 'hourly_max_pipeline'
PRED = ROOT / 'outputs' / 'predictions'
EXT = ROOT / 'data' / 'external'
ORDER = ['01-1_데이터_구조와_품질진단', '01-2_전처리와_검증전략', '02-1_사전학습모델_Chronos2',
         '02-1b_딥러닝_통계모델_예측', '02-1c_외부자료_Chronos2_미세조정',
         '02-2_15분_예측모델_비교와_선정', '02-3_피크기준_경보와_오류분석', '02-4_피크위험확률_모델비교',
         '03-1_실험모델_영향요인과_오류분석', '04-1_피크저감_운영안', '04-2_피크관리_도우미', '06_테스트_예측결과_생성']
CHRONOS_SAVED = ['chronos_day_ahead.csv', 'chronos_hour_ahead.csv', 'chronos_hour_q21.csv',
                 'extra_day_ahead.csv', 'extra_hour_ahead.csv', 'chronos_ft_hour_ahead.csv']


def use_this_python():
    """jupyter가 이 파이썬으로 커널을 띄우도록, 이 파이썬을 가리키는 커널 설정을 임시 폴더에 만들어 맨 앞에 둔다."""
    spec = Path(tempfile.mkdtemp()) / 'kernels' / 'python3'
    spec.mkdir(parents=True)
    (spec / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Python 3 (ipykernel)', 'language': 'python'}), encoding='utf-8')
    os.environ['JUPYTER_PATH'] = os.pathsep.join(p for p in [str(spec.parents[1]), os.environ.get('JUPYTER_PATH', '')] if p)


def read_tables(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob('*.csv'))}


def same_values(a, b):
    """두 표의 글자가 다를 때, 숫자 칸은 상대오차 1e-9 안이면 같다고 본다(나머지 칸은 글자 그대로 비교)."""
    ra = list(csv.reader(io.StringIO(a.decode('utf-8-sig'))))
    rb = list(csv.reader(io.StringIO(b.decode('utf-8-sig'))))
    if len(ra) != len(rb) or any(len(x) != len(y) for x, y in zip(ra, rb)):
        return False
    for x, y in zip(ra, rb):
        for u, v in zip(x, y):
            if u == v:
                continue
            try:
                if not math.isclose(float(u), float(v), rel_tol=1e-9, abs_tol=1e-12):
                    return False
            except ValueError:
                return False
    return True


def execute(path):
    cmd = [sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute', '--inplace',
           '--ExecutePreprocessor.kernel_name=python3', '--ExecutePreprocessor.timeout=-1', path.name]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=path.parent, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                       encoding='utf-8', errors='replace')
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        sys.exit(f'실행 실패: {path.name} (위 오류 참고)')
    return round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-chronos', action='store_true', help='02-1·02-1b·02-1c(저장된 예측)도 다시 계산')
    ap.add_argument('--appendix', action='store_true', help='부록 파이프라인 노트북도 다시 실행')
    ap.add_argument('--check', action='store_true', help='실행 전후 결과표 비교')
    a = ap.parse_args()
    os.environ.update({'PYTHONHASHSEED': '0', 'PYTHONIOENCODING': 'utf-8'})
    os.environ.setdefault('N_THREADS', '4')
    use_this_python()
    before = {'main': read_tables(ROOT / 'outputs' / 'tables'), 'appendix': read_tables(APPX / 'outputs' / 'tables')}
    start, times = time.time(), {}
    for name in ORDER:
        if name.startswith('02-1'):
            skip = None
            if not a.with_chronos:
                skip = '저장된 예측 사용'
            elif name.startswith('02-1c') and not EXT.exists():
                skip = 'data/external 없음, 저장된 예측 사용'
            if skip:
                if not all((PRED / f).exists() for f in CHRONOS_SAVED):
                    sys.exit('저장된 예측(Chronos-2·딥러닝·통계·미세조정)이 없습니다. --with-chronos로 다시 실행하세요.')
                print(f'건너뜀: {name} ({skip})', flush=True)
                continue
        times[name] = execute(NB / f'{name}.ipynb')
        print(f'완료: {name} ({times[name]:.0f}초)', flush=True)
    if a.appendix:
        for name in ['01_학습_파이프라인', '02_AI예측모델개발', '03_AI_확장실험']:
            times[f'appendix/{name}'] = execute(APPX / 'notebooks' / f'{name}.ipynb')
            print(f'완료: 부록 {name} ({times[f"appendix/{name}"]:.0f}초)', flush=True)
    summary = {'notebook_sec': times, 'total_sec': round(time.time() - start, 1), 'with_chronos': a.with_chronos,
               'appendix': a.appendix}
    if a.check:
        after = {'main': read_tables(ROOT / 'outputs' / 'tables'), 'appendix': read_tables(APPX / 'outputs' / 'tables')}
        for key in before:
            diff = sorted(k for k in before[key] if k in after[key] and before[key][k] != after[key][k])
            changed = [k for k in diff if not same_values(before[key][k], after[key][k])]
            float_only = [k for k in diff if k not in changed]
            added = sorted(set(after[key]) - set(before[key]))
            summary[f'check_{key}'] = {'compared': len(before[key]), 'changed': changed, 'float_only': float_only, 'new': added}
            print(f'[비교] {key}: 표 {len(before[key])}개 중 값이 달라진 표 {len(changed)}개, 부동소수점 차이만 있는 표 '
                  f'{len(float_only)}개, 새 표 {len(added)}개' + (f' → 달라진 표 {changed}' if changed else ''))
    sys.path.insert(0, str(APPX))
    import power_utils as pu
    summary['env'] = pu.get_env_info()
    (ROOT / 'outputs' / 'run_all_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"전체 완료: {summary['total_sec'] / 60:.1f}분 → outputs/run_all_summary.json", flush=True)
    try:
        import pytest  # noqa: F401
    except ImportError:
        print('pytest가 없어 tests/를 건너뜁니다(pip install pytest).')
        return
    subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider', '-W', 'ignore::FutureWarning'],
                   cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
