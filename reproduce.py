"""Verify or exactly replay the released MotionTrend checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--verify-only', action='store_true', help='CPU-only artifact and numerical verification')
    parser.add_argument('--data', type=Path, help='External original full567 NPZ directory')
    parser.add_argument('--output', type=Path, help='New evaluation directory outside code/data')
    args = parser.parse_args()
    if not args.verify_only and (args.data is None or args.output is None):
        parser.error('Supply --data and --output, or use --verify-only')
    if args.verify_only and (args.data is not None or args.output is not None):
        parser.error('--verify-only does not take --data or --output')
    root = Path(__file__).resolve().parent
    from tools.verify_release import verify_release
    result = verify_release(root)
    if args.verify_only:
        print(json.dumps(result, ensure_ascii=False))
        return 0
    from tools.evaluate import replay
    replay(root, args.data, args.output)
    verify_release(root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
