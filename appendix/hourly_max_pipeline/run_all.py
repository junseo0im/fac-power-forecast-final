"""전체 실행·재현 스크립트: 노트북 01(학습 파이프라인) → 02(결과·해석) → 03(AI 확장 실험)을 순서대로 실행한다.

사용법(프로젝트 루트에서, .venv 또는 conda 환경의 파이썬으로):
    python run_all.py --mode full             # 저장된 결과로 두 노트북을 처음부터 끝까지 실행(Restart & Run All). 약 1분
    python run_all.py --mode fast             # outputs/를 outputs_fast/로 복사해 그 위에서 점검(공식 결과는 그대로)
    python run_all.py --mode full --retrain   # 노트북 01을 RETRAIN=True로 실행해 모든 모델을 다시 학습(CPU 기준 하루 이상)

- 노트북은 papermill로 실행한다. 01에는 RETRAIN, 02에는 MODE(fast는 기준선을 최근 3 fold로만 다시 계산)를 넣는다.
- 총 실행 시간과 환경 정보는 outputs/run_all_{mode}.json에 저장한다.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / 'notebooks'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fast', 'full'], default='full')
    ap.add_argument('--retrain', action='store_true', help='노트북 01에서 모든 모델을 다시 학습(full 모드 전용)')
    args = ap.parse_args()
    if args.retrain and args.mode == 'fast':
        raise SystemExit('--retrain은 full 모드에서만 지원합니다(fast는 노트북 점검용).')
    t0 = time.time()
    os.environ.update({'PYTHONHASHSEED': '0', 'PYTHONIOENCODING': 'utf-8'})
    out_dir = ROOT / 'outputs'
    if args.mode == 'fast':                        # 공식 결과를 건드리지 않도록 복사본에서 실행
        out_dir = ROOT / 'outputs_fast'
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(ROOT / 'outputs', out_dir)
        os.environ['POWER_OUT'] = str(out_dir)

    import papermill as pm
    times = {}
    for name, params in [('01_학습_파이프라인.ipynb', {'RETRAIN': args.retrain}),
                         ('02_AI예측모델개발.ipynb', {'MODE': args.mode}),
                         ('03_AI_확장실험.ipynb', {})]:
        target = NOTEBOOKS / name if args.mode == 'full' else out_dir / name
        t1 = time.time()
        pm.execute_notebook(str(NOTEBOOKS / name), str(target), parameters=params, cwd=str(NOTEBOOKS),
                            kernel_name='python3', progress_bar=False)
        times[name] = round(time.time() - t1, 1)
        print(f'{name}: {times[name] / 60:.1f}분', flush=True)

    sys.path.insert(0, str(ROOT))
    import power_utils as pu
    info = {'mode': args.mode, 'retrain': args.retrain, 'notebook_sec': times, 'total_sec': round(time.time() - t0, 1),
            'env': pu.get_env_info()}
    (out_dir / f'run_all_{args.mode}.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"완료: 전체 {info['total_sec'] / 60:.1f}분 → {out_dir / f'run_all_{args.mode}.json'}")


if __name__ == '__main__':
    main()
