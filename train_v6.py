"""
TraFix v6 — Training Runner
=============================
Runs Stage 1 → Stage 2 → Stage 3 in sequence.
Each stage checks for its prerequisite checkpoints before starting.

Usage:
  python train_v6.py                          # full pipeline (all 3 stages)
  python train_v6.py --stage 1                # Stage 1 only
  python train_v6.py --stage 2                # Stage 2 only (needs stage1 done)
  python train_v6.py --stage 3                # Stage 3 only (needs stage1+2 done)
  python train_v6.py --stage 3 --resume trafix_v6/checkpoints/stage3_ep400.pt

  # Override default episode counts:
  python train_v6.py --s1-episodes 300 --s2-episodes 400 --s3-episodes 2000

  # Run without SUMO GUI (headless, faster):
  python train_v6.py  (GUI is off by default)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE   = Path(__file__).resolve().parent
PYTHON = sys.executable

CHECKPOINTS = HERE / "trafix_v6" / "checkpoints"
S1_CKPT = CHECKPOINTS / "stage1_gru.pt"
S2_GAT  = CHECKPOINTS / "stage2_gatconv.pt"
S2_TRK  = CHECKPOINTS / "stage2_trunk.pt"
S3_CKPT = CHECKPOINTS / "trafix_v6_final.pt"


def _header(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _run(cmd: list, label: str) -> int:
    print(f"\n[train_v6] Running: {' '.join(str(c) for c in cmd)}\n")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(HERE))
    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    hrs,  mins = divmod(mins, 60)
    duration = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"
    if result.returncode == 0:
        print(f"\n[train_v6] {label} completed in {duration}")
    else:
        print(f"\n[train_v6] {label} FAILED (exit code {result.returncode})")
    return result.returncode


def stage1(args):
    _header("Stage 1 — GRU Temporal Encoder Pretraining")
    print(f"  Episodes : {args.s1_episodes}")
    print(f"  LR       : {args.s1_lr}")
    print(f"  Output   : {S1_CKPT}")

    cmd = [
        PYTHON, "trafix_v6/stage1_pretrain_gru.py",
        "--episodes", str(args.s1_episodes),
        "--lr",       str(args.s1_lr),
    ]
    if args.gui:
        cmd.append("--gui")
    return _run(cmd, "Stage 1")


def stage2(args):
    _header("Stage 2 — GATConv + Trunk Pretraining")
    if not S1_CKPT.exists():
        print(f"[train_v6] ERROR: Stage 1 checkpoint not found: {S1_CKPT}")
        print("[train_v6] Run Stage 1 first.")
        return 1

    print(f"  Episodes         : {args.s2_episodes}")
    print(f"  Offpeak-only     : {args.s2_offpeak_episodes}")
    print(f"  LR               : {args.s2_lr}")
    print(f"  Output           : {S2_GAT}, {S2_TRK}")

    cmd = [
        PYTHON, "trafix_v6/stage2_pretrain_gatconv.py",
        "--episodes",         str(args.s2_episodes),
        "--offpeak-episodes", str(args.s2_offpeak_episodes),
        "--lr",               str(args.s2_lr),
    ]
    if args.gui:
        cmd.append("--gui")
    return _run(cmd, "Stage 2")


def stage3(args):
    _header("Stage 3 — Full PPO Training")
    if not args.resume:
        missing = []
        if not S1_CKPT.exists(): missing.append(str(S1_CKPT))
        if not S2_GAT.exists():  missing.append(str(S2_GAT))
        if not S2_TRK.exists():  missing.append(str(S2_TRK))
        if missing:
            print("[train_v6] ERROR: Missing prerequisites:")
            for m in missing: print(f"  {m}")
            return 1

    print(f"  Episodes         : {args.s3_episodes}")
    print(f"  Freeze episodes  : {args.s3_freeze}")
    print(f"  Entropy coef     : {args.s3_entropy}")
    print(f"  LR               : {args.s3_lr}")
    print(f"  Output           : {S3_CKPT}")
    if args.resume:
        print(f"  Resuming from    : {args.resume}")

    cmd = [
        PYTHON, "trafix_v6/stage3_train_ppo.py",
        "--episodes",       str(args.s3_episodes),
        "--freeze-episodes", str(args.s3_freeze),
        "--entropy-coef",   str(args.s3_entropy),
        "--lr",             str(args.s3_lr),
        "--lr-min",         str(args.s3_lr_min),
    ]
    if args.resume:
        cmd += ["--resume", args.resume]
    if args.gui:
        cmd.append("--gui")
    return _run(cmd, "Stage 3")


def main():
    parser = argparse.ArgumentParser(
        description="TraFix v6 — sequential training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None,
                        help="Run a single stage only (default: run all 3 in sequence)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Stage 3 only: path to checkpoint to resume from")
    parser.add_argument("--gui", action="store_true",
                        help="Open SUMO GUI (slower, for visual debugging)")

    # Stage 1
    g1 = parser.add_argument_group("Stage 1 — GRU pretraining")
    g1.add_argument("--s1-episodes", type=int,   default=300)
    g1.add_argument("--s1-lr",       type=float, default=1e-3)

    # Stage 2
    g2 = parser.add_argument_group("Stage 2 — GATConv + trunk pretraining")
    g2.add_argument("--s2-episodes",         type=int,   default=400)
    g2.add_argument("--s2-offpeak-episodes", type=int,   default=200,
                    help="Episodes trained on OFFPEAK only before curriculum mix")
    g2.add_argument("--s2-lr",               type=float, default=3e-4)

    # Stage 3
    g3 = parser.add_argument_group("Stage 3 — Full PPO")
    g3.add_argument("--s3-episodes",  type=int,   default=2000)
    g3.add_argument("--s3-freeze",    type=int,   default=100,
                    help="Episodes to freeze pretrained encoders")
    g3.add_argument("--s3-entropy",   type=float, default=0.01,
                    help="Entropy coefficient (higher prevents collapse over 6 phases)")
    g3.add_argument("--s3-lr",        type=float, default=3e-4)
    g3.add_argument("--s3-lr-min",    type=float, default=1e-5)

    args = parser.parse_args()

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    _header("TraFix v6 — Training Pipeline")
    print(f"  Python    : {PYTHON}")
    print(f"  Work dir  : {HERE}")
    print(f"  Checkpoints: {CHECKPOINTS}")

    if args.stage == 1:
        sys.exit(stage1(args))

    elif args.stage == 2:
        sys.exit(stage2(args))

    elif args.stage == 3:
        sys.exit(stage3(args))

    else:
        # Full pipeline
        total_start = time.time()

        rc = stage1(args)
        if rc != 0:
            print("[train_v6] Pipeline aborted at Stage 1.")
            sys.exit(rc)

        rc = stage2(args)
        if rc != 0:
            print("[train_v6] Pipeline aborted at Stage 2.")
            sys.exit(rc)

        rc = stage3(args)
        if rc != 0:
            print("[train_v6] Pipeline aborted at Stage 3.")
            sys.exit(rc)

        total = time.time() - total_start
        hrs, rem = divmod(int(total), 3600)
        mins, secs = divmod(rem, 60)
        _header("Pipeline Complete")
        print(f"  Total training time : {hrs}h {mins}m {secs}s")
        print(f"  Final model         : {S3_CKPT}")
        print()
        print("  Next steps:")
        print("    python trafix_v6/eval_stage3.py --scenarios 50 --greedy")
        print("    python baslat.py --model v6")


if __name__ == "__main__":
    main()
