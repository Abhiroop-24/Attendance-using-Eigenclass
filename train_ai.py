#!/usr/bin/env python3
"""Convenience script to train the EigenClass attendance model."""

from __future__ import annotations

import argparse
from pathlib import Path

from eigenclass_attendance import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train EigenClass model")
    parser.add_argument("--dataset", default="dataset", help="Dataset root path")
    parser.add_argument("--model", default="models/eigenclass_model.npz", help="Output model path")
    parser.add_argument("--width", type=int, default=64, help="Resize width")
    parser.add_argument("--height", type=int, default=64, help="Resize height")
    parser.add_argument("-k", type=int, default=50, help="PCA components")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model, sample_count, student_count = train_model(
        dataset_dir=Path(args.dataset),
        model_path=Path(args.model),
        image_size=(args.width, args.height),
        k=args.k,
    )

    print("Training complete")
    print(f"Samples: {sample_count}")
    print(f"Unique students: {student_count}")
    print(f"Image size: {model.image_size[0]}x{model.image_size[1]}")
    print(f"Components (k): {model.face_space.shape[1]}")
    print(f"Auto threshold: {model.threshold:.6f}")
    print(f"Model saved: {args.model}")


if __name__ == "__main__":
    main()
