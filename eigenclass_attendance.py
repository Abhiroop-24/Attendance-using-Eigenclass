#!/usr/bin/env python3
"""EIGENCLASS: Automated attendance system using an Eigenfaces (PCA) pipeline.

Directory layout expected:

project_root/
  dataset/
    Student_A/
      img1.jpg
      img2.jpg
    Student_B/
      img1.jpg
  models/
  attendance/

Usage examples:
  python eigenclass_attendance.py train --dataset dataset --model models/eigenclass_model.npz
  python eigenclass_attendance.py recognize --image new_face.jpg --model models/eigenclass_model.npz --mark
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


@dataclass
class ModelArtifacts:
    mean_face: np.ndarray        # shape: (d,)
    face_space: np.ndarray       # shape: (d, k)
    train_weights: np.ndarray    # shape: (n, k)
    labels: np.ndarray           # shape: (n,)
    image_shape: tuple[int, int]
    threshold: float


def _iter_images(folder: Path) -> Iterable[Path]:
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield p


def _load_grayscale_vector(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def load_training_data(dataset_dir: Path, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    samples: list[np.ndarray] = []
    labels: list[str] = []

    for student_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        student_name = student_dir.name
        student_images = list(_iter_images(student_dir))
        for img_path in student_images:
            samples.append(_load_grayscale_vector(img_path, size))
            labels.append(student_name)

    if not samples:
        raise ValueError(
            "No training images found. Put images under dataset/<student_name>/*.jpg"
        )

    return np.vstack(samples), np.asarray(labels)


def compute_eigenfaces(
    X: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mean face, centered matrix, eigen info, selected face space, and projections."""
    n_samples, d = X.shape
    if n_samples < 2:
        raise ValueError("Need at least 2 training images to build covariance matrix.")

    mean_face = X.mean(axis=0)
    A = X - mean_face

    # Covariance in pixel space: C = (A^T A)/(n-1), shape (d, d)
    C = (A.T @ A) / (n_samples - 1)

    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    k = max(1, min(k, eigvecs.shape[1]))
    face_space = eigvecs[:, :k]

    # Numerical guard: normalize each eigenface vector.
    norms = np.linalg.norm(face_space, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    face_space = face_space / norms

    train_weights = A @ face_space
    return mean_face, A, eigvals, eigvecs, face_space, train_weights


def estimate_threshold(train_weights: np.ndarray, labels: np.ndarray) -> float:
    """Estimate a recognition threshold using genuine vs impostor nearest distances."""
    n = train_weights.shape[0]
    dists = np.linalg.norm(train_weights[:, None, :] - train_weights[None, :, :], axis=2)

    genuine: list[float] = []
    impostor: list[float] = []

    for i in range(n):
        same = np.where(labels == labels[i])[0]
        diff = np.where(labels != labels[i])[0]

        same = same[same != i]
        if same.size:
            genuine.append(float(np.min(dists[i, same])))
        if diff.size:
            impostor.append(float(np.min(dists[i, diff])))

    if genuine and impostor:
        return float((np.mean(genuine) + np.mean(impostor)) / 2.0)
    if genuine:
        return float(np.max(genuine) * 1.15)

    # Fallback for tiny datasets (e.g., one image per student).
    return float(np.percentile(dists[dists > 0], 15)) if np.any(dists > 0) else 0.5


def save_model(path: Path, model: ModelArtifacts) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mean_face=model.mean_face,
        face_space=model.face_space,
        train_weights=model.train_weights,
        labels=model.labels,
        image_h=model.image_shape[0],
        image_w=model.image_shape[1],
        threshold=model.threshold,
    )


def load_model(path: Path) -> ModelArtifacts:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    data = np.load(path, allow_pickle=False)
    return ModelArtifacts(
        mean_face=data["mean_face"],
        face_space=data["face_space"],
        train_weights=data["train_weights"],
        labels=data["labels"],
        image_shape=(int(data["image_h"]), int(data["image_w"])),
        threshold=float(data["threshold"]),
    )


def recognize_face(model: ModelArtifacts, image_path: Path) -> tuple[str, float, bool]: #this converts an image to vector and projects into face space
    x = _load_grayscale_vector(image_path, model.image_shape)
    w = (x - model.mean_face) @ model.face_space
    distances = np.linalg.norm(model.train_weights - w, axis=1)
    i = int(np.argmin(distances))
    best_distance = float(distances[i]) 
    predicted_label = str(model.labels[i])
    recognized = best_distance <= model.threshold
    return (predicted_label if recognized else "UNKNOWN", best_distance, recognized)#picks teh closest match and checks if its recongnized or not


def mark_attendance(attendance_csv: Path, student: str, distance: float, image_path: Path, recognized: bool) -> None:
    attendance_csv.parent.mkdir(parents=True, exist_ok=True)

    now = dt.datetime.now()
    status = "PRESENT" if recognized else "UNKNOWN"

    # Prevent duplicate PRESENT entries for the same student on the same date.
    today = now.date().isoformat()
    existing_rows: list[dict[str, str]] = []
    already_present_today = False

    if attendance_csv.exists():
        with attendance_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                if (
                    row.get("date") == today
                    and row.get("student") == student
                    and row.get("status") == "PRESENT"
                ):
                    already_present_today = True

    if recognized and already_present_today:
        return

    header = ["date", "time", "student", "status", "distance", "image"]
    write_header = not attendance_csv.exists()

    with attendance_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(
            [
                now.date().isoformat(),
                now.time().strftime("%H:%M:%S"),
                student,
                status,
                f"{distance:.6f}",
                str(image_path),
            ]
        )


def run_train(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset)
    model_path = Path(args.model)
    image_size = (args.width, args.height)

    X, labels = load_training_data(dataset_dir, image_size)
    mean_face, _A, _eigvals, _eigvecs, face_space, train_weights = compute_eigenfaces(X, args.k)
    threshold = estimate_threshold(train_weights, labels)

    model = ModelArtifacts(
        mean_face=mean_face,
        face_space=face_space,
        train_weights=train_weights,
        labels=labels,
        image_shape=image_size,
        threshold=threshold,
    )
    save_model(model_path, model)

    print("Training complete")
    print(f"Samples: {len(labels)}")
    print(f"Unique students: {len(set(labels.tolist()))}")
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    print(f"Components (k): {model.face_space.shape[1]}")
    print(f"Auto threshold: {model.threshold:.6f}")
    print(f"Model saved: {model_path}")


def run_recognize(args: argparse.Namespace) -> None:
    model = load_model(Path(args.model))
    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    label, distance, recognized = recognize_face(model, image_path)
    print(f"Prediction: {label}")
    print(f"Distance: {distance:.6f}")
    print(f"Threshold: {model.threshold:.6f}")
    print(f"Recognized: {recognized}")

    if args.mark:
        mark_attendance(Path(args.attendance), label, distance, image_path, recognized)
        print(f"Attendance updated: {args.attendance}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EIGENCLASS attendance system using linear algebra + Eigenfaces"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Train model from dataset")
    p_train.add_argument("--dataset", default="dataset", help="Path to dataset root")
    p_train.add_argument("--model", default="models/eigenclass_model.npz", help="Output model path")
    p_train.add_argument("--width", type=int, default=64, help="Resize width")
    p_train.add_argument("--height", type=int, default=64, help="Resize height")
    p_train.add_argument("-k", type=int, default=50, help="Number of PCA components")
    p_train.set_defaults(func=run_train)

    p_rec = subparsers.add_parser("recognize", help="Recognize one face image")
    p_rec.add_argument("--image", required=True, help="Input face image path")
    p_rec.add_argument("--model", default="models/eigenclass_model.npz", help="Path to trained model")
    p_rec.add_argument("--mark", action="store_true", help="Write attendance entry")
    p_rec.add_argument(
        "--attendance",
        default="attendance/attendance.csv",
        help="Attendance CSV path",
    )
    p_rec.set_defaults(func=run_recognize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
