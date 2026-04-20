#!/usr/bin/env python3
"""One-command onboarding + live attendance launcher."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch final attendance setup flow")
    parser.add_argument("--python", default=sys.executable, help="Python executable path")
    parser.add_argument("--script", default="eigenclass_attendance.py", help="Main script path")
    parser.add_argument("--dataset", default="dataset", help="Dataset root")
    parser.add_argument("--model", default="models/eigenclass_model.npz", help="Model file path")
    parser.add_argument("--attendance", default="attendance/attendance.csv", help="Attendance CSV path")
    parser.add_argument("--min-samples", type=int, default=5, help="Minimum images during enrollment")
    parser.add_argument("--max-samples", type=int, default=8, help="Maximum images during enrollment")
    parser.add_argument("--camera", type=int, default=0, help="Laptop camera index")
    parser.add_argument("--width", type=int, default=64, help="Training width")
    parser.add_argument("--height", type=int, default=64, help="Training height")
    parser.add_argument("-k", type=int, default=50, help="PCA components")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    script_path = Path(args.script)
    if not script_path.exists():
        raise FileNotFoundError(f"Main script not found: {script_path}")

    print("=== EIGENCLASS Final Setup ===")
    name = input("Enter student name: ").strip()
    srn = input("Enter student SRN (example CS024): ").strip().upper()

    if not name or not srn:
        raise ValueError("Both name and SRN are required.")

    enroll_cmd = [
        args.python,
        str(script_path),
        "enroll",
        "--name",
        name,
        "--srn",
        srn,
        "--dataset",
        args.dataset,
        "--min-samples",
        str(args.min_samples),
        "--max-samples",
        str(args.max_samples),
        "--camera",
        str(args.camera),
        "--model",
        args.model,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "-k",
        str(args.k),
    ]
    _run(enroll_cmd)

    print("\nEnrollment and training complete.")
    print("Starting live attendance now. Press 'q' in camera window to stop.\n")

    live_cmd = [
        args.python,
        str(script_path),
        "live",
        "--model",
        args.model,
        "--attendance",
        args.attendance,
        "--camera",
        str(args.camera),
        "--mark",
    ]
    _run(live_cmd)


if __name__ == "__main__":
    main()
