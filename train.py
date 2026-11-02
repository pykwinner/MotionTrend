"""Train the released MoTrend recipe without requiring Torch for --help."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--data', type=Path, required=True, help='Original full 567-file NPZ directory')
    parser.add_argument('--output', type=Path, required=True, help='New output directory, or owned run with --resume')
    parser.add_argument('--resume', action='store_true', help='Continue after the latest closed epoch; do not repeat completed epochs')
    parser.add_argument('--preflight-only', action='store_true', help='Check native gradients and fixed inputs, retaining no updates')
    args = parser.parse_args()
    if args.resume and args.preflight_only:
        parser.error('--resume and --preflight-only cannot be combined')
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    from training.runner import run
    return run(args, Path(__file__).resolve().parent)


if __name__ == '__main__':
    raise SystemExit(main())
