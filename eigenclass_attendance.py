#!/usr/bin/env python3
"""EIGENCLASS: Smart attendance with Eigenfaces + webcam registration + Ollama math.

Usage examples:
  python eigenclass_attendance.py register --name "Abhiroop" --srn CS024 --samples 5 --train-after
    python eigenclass_attendance.py enroll --srn CS024 --name "Abhiroop"
  python eigenclass_attendance.py train --dataset dataset --model models/eigenclass_model.npz
  python eigenclass_attendance.py live --model models/eigenclass_model.npz --mark
  python eigenclass_attendance.py math --query "Differentiate x^3 + 5x"
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import operator
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only when OpenCV is missing
    cv2 = None


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
REGISTRY_FILENAME = "student_registry.csv"


@dataclass
class ModelArtifacts:
    mean_face: np.ndarray        # shape: (d,)
    face_space: np.ndarray       # shape: (d, k)
    train_weights: np.ndarray    # shape: (n, k)
    srn_labels: np.ndarray       # shape: (n,)
    name_labels: np.ndarray      # shape: (n,)
    image_size: tuple[int, int]  # (width, height)
    threshold: float
    margin_threshold: float


@dataclass
class StudentPrediction:
    srn: str
    name: str
    distance: float
    recognized: bool


@dataclass
class LiveTrack:
    bbox: tuple[int, int, int, int]
    last_seen_frame: int
    votes: list[str]
    stable_srn: str | None = None
    stable_name: str | None = None
    stable_distance: float = 0.0
    stable_streak: int = 0
    marked: bool = False


def _iter_images(folder: Path) -> Iterable[Path]:
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield p


def _load_grayscale_vector(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    return _prepare_face_vector(arr, size=size, use_cuda_preprocess=False)


def _cuda_available() -> bool:
    if cv2 is None or not hasattr(cv2, "cuda"):
        return False
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def _prepare_face_vector(
    gray_face: np.ndarray,
    size: tuple[int, int],
    use_cuda_preprocess: bool,
) -> np.ndarray:
    _require_cv2()

    working = gray_face
    if working.dtype != np.uint8:
        working = np.clip(working, 0, 255).astype(np.uint8)

    if use_cuda_preprocess and _cuda_available():
        try:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(working)
            gpu_resized = cv2.cuda.resize(gpu, size, interpolation=cv2.INTER_AREA)
            working = gpu_resized.download()
        except Exception:
            working = cv2.resize(working, size, interpolation=cv2.INTER_AREA)
    else:
        working = cv2.resize(working, size, interpolation=cv2.INTER_AREA)

    # Lighting normalization improves stability across classroom conditions.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(working)
    arr = norm.astype(np.float32) / 255.0
    return arr.reshape(-1)


def _normalize_srn(value: str) -> str:
    srn = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9_-]{3,32}", srn):
        raise ValueError(
            "Invalid SRN format. Use letters/numbers and optional '-' or '_' (3-32 chars)."
        )
    return srn


def _prompt_non_empty(prompt_text: str, default: str | None = None) -> str:
    while True:
        value = input(prompt_text).strip()
        if value:
            return value
        if default is not None:
            return default
        print("Input cannot be empty.")


def _validate_production_sample_window(min_samples: int, max_samples: int) -> None:
    if min_samples < 5 or max_samples > 8 or min_samples > max_samples:
        raise ValueError("For production enrollment use a 5-8 image window with min <= max.")


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for webcam features. Install with: pip install opencv-python"
        )


def _get_face_detector() -> object:
    _require_cv2()
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
    return detector


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    union = (aw * ah) + (bw * bh) - inter
    return (inter / union) if union > 0 else 0.0


def _stable_vote_from_track(
    track: LiveTrack,
    min_votes: int,
    vote_ratio: float,
) -> tuple[str | None, int, int]:
    non_unknown = [v for v in track.votes if v != "UNKNOWN"]
    if not non_unknown:
        return None, 0, len(track.votes)

    counts: dict[str, int] = {}
    for srn in non_unknown:
        counts[srn] = counts.get(srn, 0) + 1

    best_srn = max(counts.items(), key=lambda it: it[1])[0]
    best_count = counts[best_srn]
    total_votes = len(track.votes)
    if best_count < min_votes:
        return None, best_count, total_votes
    if total_votes == 0 or (best_count / total_votes) < vote_ratio:
        return None, best_count, total_votes
    return best_srn, best_count, total_votes


def _registry_path(dataset_dir: Path) -> Path:
    return dataset_dir / REGISTRY_FILENAME


def load_registry(dataset_dir: Path) -> dict[str, dict[str, str]]:
    registry_csv = _registry_path(dataset_dir)
    rows_by_folder: dict[str, dict[str, str]] = {}

    if not registry_csv.exists():
        return rows_by_folder

    with registry_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = (row.get("folder") or row.get("srn") or "").strip()
            if not folder:
                continue
            rows_by_folder[folder] = {
                "srn": (row.get("srn") or folder).strip(),
                "name": (row.get("name") or folder).strip(),
                "folder": folder,
            }

    return rows_by_folder


def upsert_registry_entry(
    dataset_dir: Path,
    srn: str,
    name: str,
    folder: str,
    sample_count: int,
) -> None:
    registry_csv = _registry_path(dataset_dir)
    now = dt.datetime.now().isoformat(timespec="seconds")

    rows: list[dict[str, str]] = []
    if registry_csv.exists():
        with registry_csv.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    updated = False
    for row in rows:
        row_srn = (row.get("srn") or "").strip().upper()
        row_folder = (row.get("folder") or "").strip()
        if row_srn == srn or row_folder == folder:
            row["srn"] = srn
            row["name"] = name
            row["folder"] = folder
            row["samples"] = str(sample_count)
            row["updated_at"] = now
            updated = True
            break

    if not updated:
        rows.append(
            {
                "srn": srn,
                "name": name,
                "folder": folder,
                "samples": str(sample_count),
                "created_at": now,
                "updated_at": now,
            }
        )

    rows.sort(key=lambda row: row.get("srn", ""))
    header = ["srn", "name", "folder", "samples", "created_at", "updated_at"]
    with registry_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def load_training_data(
    dataset_dir: Path,
    size: tuple[int, int],
    use_cuda_preprocess: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    registry = load_registry(dataset_dir)
    samples: list[np.ndarray] = []
    srn_labels: list[str] = []
    name_labels: list[str] = []

    for student_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        meta = registry.get(student_dir.name)
        student_srn = (meta["srn"] if meta else student_dir.name).strip()
        student_name = (meta["name"] if meta else student_dir.name).strip()

        for img_path in _iter_images(student_dir):
            img = Image.open(img_path).convert("L")
            arr = np.asarray(img, dtype=np.uint8)
            samples.append(_prepare_face_vector(arr, size=size, use_cuda_preprocess=use_cuda_preprocess))
            srn_labels.append(student_srn)
            name_labels.append(student_name)

    if not samples:
        raise ValueError(
            "No training images found. Put images under dataset/<SRN>/*.jpg or dataset/<name>/*.jpg"
        )

    return np.vstack(samples), np.asarray(srn_labels), np.asarray(name_labels)


def compute_eigenfaces(
    X: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mean face, centered matrix, eigen info, selected face space, and projections."""
    n_samples, _d = X.shape
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

    norms = np.linalg.norm(face_space, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    face_space = face_space / norms

    train_weights = A @ face_space
    return mean_face, A, eigvals, eigvecs, face_space, train_weights


def estimate_threshold(train_weights: np.ndarray, labels: np.ndarray) -> float:
    """Estimate a threshold from nearest same-student vs different-student distances."""
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

    return float(np.percentile(dists[dists > 0], 15)) if np.any(dists > 0) else 0.5


def estimate_margin_threshold(train_weights: np.ndarray, labels: np.ndarray) -> float:
    n = train_weights.shape[0]
    if n < 3:
        return 0.0

    dists = np.linalg.norm(train_weights[:, None, :] - train_weights[None, :, :], axis=2)
    margins: list[float] = []

    for i in range(n):
        same = np.where(labels == labels[i])[0]
        diff = np.where(labels != labels[i])[0]
        same = same[same != i]
        if same.size == 0 or diff.size == 0:
            continue

        best_same = float(np.min(dists[i, same]))
        best_diff = float(np.min(dists[i, diff]))
        margins.append(best_diff - best_same)

    if not margins:
        return 0.0

    # Lower quartile margin keeps matches conservative but not overly strict.
    return float(max(0.0, np.percentile(margins, 25)))


def save_model(path: Path, model: ModelArtifacts) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mean_face=model.mean_face,
        face_space=model.face_space,
        train_weights=model.train_weights,
        srn_labels=model.srn_labels,
        name_labels=model.name_labels,
        labels=model.srn_labels,
        image_h=model.image_size[0],
        image_w=model.image_size[1],
        threshold=model.threshold,
        margin_threshold=model.margin_threshold,
    )


def load_model(path: Path) -> ModelArtifacts:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    data = np.load(path, allow_pickle=False)
    srn_labels = data["srn_labels"] if "srn_labels" in data.files else data["labels"]
    name_labels = data["name_labels"] if "name_labels" in data.files else srn_labels

    return ModelArtifacts(
        mean_face=data["mean_face"],
        face_space=data["face_space"],
        train_weights=data["train_weights"],
        srn_labels=srn_labels,
        name_labels=name_labels,
        image_size=(int(data["image_h"]), int(data["image_w"])),
        threshold=float(data["threshold"]),
        margin_threshold=float(data["margin_threshold"]) if "margin_threshold" in data.files else 0.0,
    )


def recognize_vector(
    model: ModelArtifacts,
    x: np.ndarray,
    distance_threshold: float | None = None,
    margin_threshold: float | None = None,
) -> StudentPrediction:
    w = (x - model.mean_face) @ model.face_space
    distances = np.linalg.norm(model.train_weights - w, axis=1)
    i = int(np.argmin(distances))
    best_distance = float(distances[i])
    top_two = np.partition(distances, 1)[:2]
    second_best = float(np.max(top_two)) if len(top_two) >= 2 else best_distance
    margin = second_best - best_distance

    active_distance_threshold = model.threshold if distance_threshold is None else distance_threshold
    active_margin_threshold = (
        model.margin_threshold if margin_threshold is None else margin_threshold
    )
    recognized = (best_distance <= active_distance_threshold) and (margin >= active_margin_threshold)

    if recognized:
        return StudentPrediction(
            srn=str(model.srn_labels[i]),
            name=str(model.name_labels[i]),
            distance=best_distance,
            recognized=True,
        )

    return StudentPrediction(srn="UNKNOWN", name="UNKNOWN", distance=best_distance, recognized=False)


def recognize_face(
    model: ModelArtifacts,
    image_path: Path,
    distance_threshold: float | None = None,
    margin_threshold: float | None = None,
) -> StudentPrediction:
    x = _load_grayscale_vector(image_path, model.image_size)
    return recognize_vector(
        model,
        x,
        distance_threshold=distance_threshold,
        margin_threshold=margin_threshold,
    )


def recognize_face_crop(
    model: ModelArtifacts,
    face_crop_gray: np.ndarray,
    distance_threshold: float | None = None,
    margin_threshold: float | None = None,
    use_cuda_preprocess: bool = False,
) -> StudentPrediction:
    x = _prepare_face_vector(face_crop_gray, model.image_size, use_cuda_preprocess=use_cuda_preprocess)
    return recognize_vector(
        model,
        x,
        distance_threshold=distance_threshold,
        margin_threshold=margin_threshold,
    )


def mark_attendance(
    attendance_csv: Path,
    prediction: StudentPrediction,
    source: str,
    allow_unknown: bool,
) -> bool:
    if not prediction.recognized and not allow_unknown:
        return False

    attendance_csv.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now()
    status = "PRESENT" if prediction.recognized else "UNKNOWN"
    today = now.date().isoformat()

    already_present_today = False
    if attendance_csv.exists():
        with attendance_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_srn = row.get("srn") or row.get("student")
                if (
                    row.get("date") == today
                    and row_srn == prediction.srn
                    and row.get("status") == "PRESENT"
                ):
                    already_present_today = True
                    break

    if prediction.recognized and already_present_today:
        return False

    header = ["date", "time", "srn", "name", "status", "distance", "source"]
    write_header = not attendance_csv.exists()

    with attendance_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(
            [
                now.date().isoformat(),
                now.time().strftime("%H:%M:%S"),
                prediction.srn,
                prediction.name,
                status,
                f"{prediction.distance:.6f}",
                source,
            ]
        )

    return True


def train_model(
    dataset_dir: Path,
    model_path: Path,
    image_size: tuple[int, int],
    k: int,
    use_cuda_preprocess: bool = False,
) -> tuple[ModelArtifacts, int, int]:
    X, srn_labels, name_labels = load_training_data(
        dataset_dir,
        image_size,
        use_cuda_preprocess=use_cuda_preprocess,
    )
    mean_face, _A, _eigvals, _eigvecs, face_space, train_weights = compute_eigenfaces(X, k)
    threshold = estimate_threshold(train_weights, srn_labels)
    margin_threshold = estimate_margin_threshold(train_weights, srn_labels)

    model = ModelArtifacts(
        mean_face=mean_face,
        face_space=face_space,
        train_weights=train_weights,
        srn_labels=srn_labels,
        name_labels=name_labels,
        image_size=image_size,
        threshold=threshold,
        margin_threshold=margin_threshold,
    )
    save_model(model_path, model)
    return model, len(srn_labels), len(set(srn_labels.tolist()))


def capture_student_faces(
    student_dir: Path,
    srn: str,
    min_samples: int,
    max_samples: int,
    camera_index: int,
    capture_size: tuple[int, int],
    cooldown_seconds: float,
    blur_threshold: float,
) -> int:
    if min_samples < 1 or max_samples < 1 or min_samples > max_samples:
        raise ValueError("Invalid sample range. Ensure 1 <= min_samples <= max_samples.")

    detector = _get_face_detector()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open laptop camera index {camera_index}.")

    existing_count = len(list(_iter_images(student_dir)))
    captured = 0
    last_capture_time = 0.0
    window_name = f"Registration - {srn} (press q to quit)"
    status_line = "Show one clear face in the frame."

    try:
        while captured < max_samples:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(80, 80),
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 200, 40), 2)

            now = time.time()
            if len(faces) == 0:
                status_line = "No face detected."
            elif len(faces) > 1:
                status_line = "Multiple faces detected. Keep only one student in frame."
            elif (now - last_capture_time) < cooldown_seconds:
                status_line = "Hold still..."
            else:
                x, y, w, h = faces[0]
                face_crop = gray[y : y + h, x : x + w]
                if face_crop.size > 0:
                    blur_score = float(cv2.Laplacian(face_crop, cv2.CV_64F).var())
                    if blur_score >= blur_threshold:
                        face = cv2.resize(face_crop, capture_size, interpolation=cv2.INTER_AREA)
                        image_index = existing_count + captured + 1
                        out_path = student_dir / f"{srn.lower()}_{image_index:02d}.jpg"
                        cv2.imwrite(str(out_path), face)
                        captured += 1
                        last_capture_time = now
                        status_line = f"Captured {captured}/{max_samples}"
                    else:
                        status_line = (
                            f"Image too blurry ({blur_score:.1f} < {blur_threshold:.1f})."
                        )

            cv2.putText(
                frame,
                f"Captured {captured}/{max_samples} (min {min_samples})",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                status_line,
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            if captured >= min_samples:
                cv2.putText(
                    frame,
                    "Press q to finish enrollment early.",
                    (12, 86),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                if captured >= min_samples:
                    break
                print(
                    f"Cannot stop yet. Need at least {min_samples} captures, currently {captured}."
                )
                continue
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured


def _extract_simple_arithmetic_expression(question: str) -> str | None:
    candidate = question.strip()
    candidate = re.sub(r"^(what\s+is|calculate|compute|evaluate)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.strip().rstrip("? ")
    if re.fullmatch(r"[0-9\s\+\-\*\/\(\)\.%\^]+", candidate):
        return candidate
    return None


def _safe_eval_expression(expression: str) -> float:
    expression = expression.replace("^", "**")

    binary_ops: dict[type[ast.AST], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }
    unary_ops: dict[type[ast.AST], Callable[[float], float]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in binary_ops:
            return float(binary_ops[type(node.op)](_eval(node.left), _eval(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
            return float(unary_ops[type(node.op)](_eval(node.operand)))
        raise ValueError("Unsupported expression")

    tree = ast.parse(expression, mode="eval")
    return _eval(tree)


def _format_numeric_result(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.10g}"


def ask_math_with_ollama(
    question: str,
    model_name: str,
    timeout_seconds: int,
    verified_result: str | None = None,
) -> str:
    verified_line = ""
    if verified_result is not None:
        verified_line = (
            f"The exact numeric answer for the arithmetic part is: {verified_result}. "
            "Your final answer must match that value exactly.\n"
        )

    prompt = (
        "You are a precise math assistant. Keep response short: steps then final answer.\n"
        f"{verified_line}"
        f"Question: {question}"
    )
    command = ["ollama", "run", model_name, prompt]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Ollama CLI not found. Install Ollama and ensure `ollama` is on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        error_details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"Ollama request failed: {error_details}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Ollama request timed out. Try increasing --timeout.") from exc

    answer = result.stdout.strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty response.")
    return answer


def run_train(args: argparse.Namespace) -> None:
    model, sample_count, student_count = train_model(
        dataset_dir=Path(args.dataset),
        model_path=Path(args.model),
        image_size=(args.width, args.height),
        k=args.k,
        use_cuda_preprocess=args.use_cuda_preprocess,
    )

    print("Training complete")
    print(f"Samples: {sample_count}")
    print(f"Unique students: {student_count}")
    print(f"Image size: {model.image_size[0]}x{model.image_size[1]}")
    print(f"Components (k): {model.face_space.shape[1]}")
    print(f"Auto threshold: {model.threshold:.6f}")
    print(f"Margin threshold: {model.margin_threshold:.6f}")
    print(f"Model saved: {args.model}")


def run_register(args: argparse.Namespace) -> None:
    _require_cv2()

    name = args.name.strip()
    if not name:
        raise ValueError("Name cannot be empty.")

    srn = _normalize_srn(args.srn)
    dataset_dir = Path(args.dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    student_dir = dataset_dir / srn
    student_dir.mkdir(parents=True, exist_ok=True)

    captured = capture_student_faces(
        student_dir=student_dir,
        srn=srn,
        min_samples=args.samples,
        max_samples=args.samples,
        camera_index=args.camera,
        capture_size=(args.capture_width, args.capture_height),
        cooldown_seconds=args.cooldown,
        blur_threshold=args.blur_threshold,
    )

    if captured < args.samples:
        raise RuntimeError(
            f"Registration stopped early. Captured {captured}/{args.samples} images."
        )

    total_images = len(list(_iter_images(student_dir)))
    upsert_registry_entry(dataset_dir, srn, name, student_dir.name, total_images)

    print("Registration complete")
    print(f"Name: {name}")
    print(f"SRN: {srn}")
    print(f"Saved folder: {student_dir}")
    print(f"Total samples for this student: {total_images}")

    if args.train_after:
        model, sample_count, student_count = train_model(
            dataset_dir=dataset_dir,
            model_path=Path(args.model),
            image_size=(args.width, args.height),
            k=args.k,
            use_cuda_preprocess=args.use_cuda_preprocess,
        )
        print("Model retrained after registration")
        print(f"Samples: {sample_count}")
        print(f"Unique students: {student_count}")
        print(f"Auto threshold: {model.threshold:.6f}")
        print(f"Margin threshold: {model.margin_threshold:.6f}")
        print(f"Model saved: {args.model}")


def run_enroll(args: argparse.Namespace) -> None:
    _require_cv2()

    dataset_dir = Path(args.dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    srn_input = args.srn.strip() if args.srn else _prompt_non_empty(
        "Enter student SRN (example CS024): "
    )
    srn = _normalize_srn(srn_input)

    if args.name:
        name = args.name.strip()
    else:
        name = _prompt_non_empty(f"Enter student name [{srn}]: ", default=srn)

    if not name:
        raise ValueError("Student name cannot be empty.")

    _validate_production_sample_window(args.min_samples, args.max_samples)

    registry = load_registry(dataset_dir)
    if srn in registry and not args.allow_update:
        raise RuntimeError(
            "SRN already exists in registry. Re-run with --allow-update to add fresh samples."
        )

    student_dir = dataset_dir / srn
    student_dir.mkdir(parents=True, exist_ok=True)

    captured = capture_student_faces(
        student_dir=student_dir,
        srn=srn,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        camera_index=args.camera,
        capture_size=(args.capture_width, args.capture_height),
        cooldown_seconds=args.cooldown,
        blur_threshold=args.blur_threshold,
    )

    if captured < args.min_samples:
        raise RuntimeError(
            f"Enrollment stopped early. Captured {captured}/{args.min_samples} required images."
        )

    total_images = len(list(_iter_images(student_dir)))
    upsert_registry_entry(dataset_dir, srn, name, student_dir.name, total_images)

    print("Enrollment complete")
    print(f"Name: {name}")
    print(f"SRN: {srn}")
    print(f"Captured this session: {captured}")
    print(f"Total samples for this student: {total_images}")

    model, sample_count, student_count = train_model(
        dataset_dir=dataset_dir,
        model_path=Path(args.model),
        image_size=(args.width, args.height),
        k=args.k,
        use_cuda_preprocess=args.use_cuda_preprocess,
    )
    print("Model retrained after enrollment")
    print(f"Samples: {sample_count}")
    print(f"Unique students: {student_count}")
    print(f"Auto threshold: {model.threshold:.6f}")
    print(f"Margin threshold: {model.margin_threshold:.6f}")
    print(f"Model saved: {args.model}")


def run_recognize(args: argparse.Namespace) -> None:
    model = load_model(Path(args.model))
    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    distance_threshold = model.threshold * args.threshold_scale
    active_margin_threshold = args.margin_threshold if args.margin_threshold > 0 else None
    prediction = recognize_face(
        model,
        image_path,
        distance_threshold=distance_threshold,
        margin_threshold=active_margin_threshold,
    )
    print(f"Prediction SRN: {prediction.srn}")
    print(f"Prediction Name: {prediction.name}")
    print(f"Distance: {prediction.distance:.6f}")
    print(f"Threshold: {distance_threshold:.6f}")
    print(f"Recognized: {prediction.recognized}")

    if args.mark:
        wrote = mark_attendance(
            attendance_csv=Path(args.attendance),
            prediction=prediction,
            source=str(image_path),
            allow_unknown=not args.skip_unknown,
        )
        if wrote:
            print(f"Attendance updated: {args.attendance}")
        else:
            print("Attendance not updated (duplicate or unknown skipped).")


def run_live(args: argparse.Namespace) -> None:
    _require_cv2()

    model = load_model(Path(args.model))
    detector = _get_face_detector()
    attendance_csv = Path(args.attendance)
    distance_threshold = model.threshold * args.threshold_scale
    margin_threshold = args.margin_threshold if args.margin_threshold > 0 else model.margin_threshold

    if args.use_cuda_preprocess and not _cuda_available():
        print("CUDA preprocessing requested but no CUDA device detected. Falling back to CPU.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open laptop camera index {args.camera}.")

    marked_this_session: set[str] = set()
    window_name = "EIGENCLASS Live Attendance (press q to quit)"
    frame_index = 0
    next_track_id = 1
    tracks: dict[int, LiveTrack] = {}
    srn_to_name: dict[str, str] = {}
    for srn, name in zip(model.srn_labels.tolist(), model.name_labels.tolist()):
        if srn not in srn_to_name:
            srn_to_name[srn] = name

    print("Live attendance started. Press 'q' to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame_index += 1

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(args.min_face, args.min_face),
            )

            used_track_ids: set[int] = set()
            for (x, y, w, h) in faces:
                face_crop = gray[y : y + h, x : x + w]
                if face_crop.size == 0:
                    continue

                prediction = recognize_face_crop(
                    model,
                    face_crop,
                    distance_threshold=distance_threshold,
                    margin_threshold=margin_threshold,
                    use_cuda_preprocess=args.use_cuda_preprocess,
                )

                bbox = (int(x), int(y), int(w), int(h))
                matched_track_id: int | None = None
                best_iou = 0.0
                for track_id, track in tracks.items():
                    if track_id in used_track_ids:
                        continue
                    iou = _bbox_iou(track.bbox, bbox)
                    if iou >= args.track_iou and iou > best_iou:
                        best_iou = iou
                        matched_track_id = track_id

                if matched_track_id is None:
                    matched_track_id = next_track_id
                    next_track_id += 1
                    tracks[matched_track_id] = LiveTrack(
                        bbox=bbox,
                        last_seen_frame=frame_index,
                        votes=[],
                    )

                used_track_ids.add(matched_track_id)
                track = tracks[matched_track_id]
                track.bbox = bbox
                track.last_seen_frame = frame_index

                vote_label = prediction.srn if prediction.recognized else "UNKNOWN"
                track.votes.append(vote_label)
                if len(track.votes) > args.vote_window:
                    track.votes = track.votes[-args.vote_window:]

                voted_srn, voted_count, total_votes = _stable_vote_from_track(
                    track,
                    min_votes=args.min_votes,
                    vote_ratio=args.vote_ratio,
                )

                if voted_srn is not None and frame_index >= args.min_track_age:
                    voted_name = srn_to_name.get(voted_srn, voted_srn)
                    if track.stable_srn == voted_srn:
                        track.stable_streak += 1
                    else:
                        track.stable_srn = voted_srn
                        track.stable_name = voted_name
                        track.stable_streak = 1
                    track.stable_distance = prediction.distance

                    if args.mark and track.stable_streak >= args.confirm_frames:
                        if (not track.marked) and (track.stable_srn not in marked_this_session):
                            final_prediction = StudentPrediction(
                                srn=track.stable_srn,
                                name=track.stable_name or track.stable_srn,
                                distance=track.stable_distance,
                                recognized=True,
                            )
                            wrote = mark_attendance(
                                attendance_csv=attendance_csv,
                                prediction=final_prediction,
                                source=f"camera:{args.camera}",
                                allow_unknown=False,
                            )
                            if wrote:
                                print(
                                    f"Marked PRESENT: {final_prediction.name} ({final_prediction.srn})"
                                )
                            marked_this_session.add(final_prediction.srn)
                            track.marked = True

                    color = (40, 200, 40)
                    label = (
                        f"{track.stable_name} ({track.stable_srn}) "
                        f"[{voted_count}/{total_votes}]"
                    )
                else:
                    track.stable_streak = max(0, track.stable_streak - 1)
                    color = (30, 30, 220)
                    label = (
                        "SCANNING "
                        f"[{voted_count}/{total_votes}]"
                        if prediction.distance <= (distance_threshold * 1.2)
                        else "UNKNOWN"
                    )

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"{label} | d={prediction.distance:.2f}",
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            # Remove stale tracks so votes from old faces do not leak into new detections.
            stale_ids = [
                track_id
                for track_id, track in tracks.items()
                if (frame_index - track.last_seen_frame) > args.track_ttl
            ]
            for track_id in stale_ids:
                tracks.pop(track_id, None)

            cv2.putText(
                frame,
                f"Threshold={distance_threshold:.2f}  Margin={margin_threshold:.2f}  Confirm={args.confirm_frames}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Tracks={len(tracks)}  VoteWin={args.vote_window}  Press q to quit",
                (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Live session finished. Unique marks in this session: {len(marked_this_session)}")


def run_math(args: argparse.Namespace) -> None:
    verified_result: str | None = None
    expression = _extract_simple_arithmetic_expression(args.query)
    if expression is not None:
        try:
            verified_value = _safe_eval_expression(expression)
            verified_result = _format_numeric_result(verified_value)
        except (ValueError, SyntaxError):
            verified_result = None

    answer = ask_math_with_ollama(
        question=args.query,
        model_name=args.ollama_model,
        timeout_seconds=args.timeout,
        verified_result=verified_result,
    )
    print(answer)
    if verified_result is not None and verified_result not in answer:
        print(f"Verified numeric answer: {verified_result}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EIGENCLASS smart attendance using linear algebra + Eigenfaces"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Train model from dataset")
    p_train.add_argument("--dataset", default="dataset", help="Path to dataset root")
    p_train.add_argument("--model", default="models/eigenclass_model.npz", help="Output model path")
    p_train.add_argument("--width", type=int, default=64, help="Resize width for training")
    p_train.add_argument("--height", type=int, default=64, help="Resize height for training")
    p_train.add_argument("-k", type=int, default=50, help="Number of PCA components")
    p_train.add_argument(
        "--use-cuda-preprocess",
        action="store_true",
        help="Try CUDA for image preprocessing (falls back to CPU if unavailable)",
    )
    p_train.set_defaults(func=run_train)

    p_register = subparsers.add_parser(
        "register",
        help="Register one student from laptop camera with face detection",
    )
    p_register.add_argument("--name", required=True, help="Student name")
    p_register.add_argument("--srn", required=True, help="Student SRN (example: CS024)")
    p_register.add_argument("--dataset", default="dataset", help="Path to dataset root")
    p_register.add_argument("--samples", type=int, default=5, help="Number of face images to capture")
    p_register.add_argument("--camera", type=int, default=0, help="Laptop camera index")
    p_register.add_argument("--capture-width", type=int, default=160, help="Saved face width")
    p_register.add_argument("--capture-height", type=int, default=160, help="Saved face height")
    p_register.add_argument("--cooldown", type=float, default=0.9, help="Seconds between captures")
    p_register.add_argument(
        "--blur-threshold",
        type=float,
        default=80.0,
        help="Minimum blur score (Laplacian variance) for a saved frame",
    )
    p_register.add_argument("--train-after", action="store_true", help="Retrain model after registration")
    p_register.add_argument(
        "--use-cuda-preprocess",
        action="store_true",
        help="Try CUDA for train-after preprocessing (falls back to CPU if unavailable)",
    )
    p_register.add_argument("--model", default="models/eigenclass_model.npz", help="Model path for --train-after")
    p_register.add_argument("--width", type=int, default=64, help="Training resize width for --train-after")
    p_register.add_argument("--height", type=int, default=64, help="Training resize height for --train-after")
    p_register.add_argument("-k", type=int, default=50, help="PCA components for --train-after")
    p_register.set_defaults(func=run_register)

    p_enroll = subparsers.add_parser(
        "enroll",
        help="Production SRN-first onboarding: capture 5-8 images and auto-train",
    )
    p_enroll.add_argument("--srn", help="Student SRN (if omitted, you will be prompted)")
    p_enroll.add_argument("--name", help="Student name (if omitted, you will be prompted)")
    p_enroll.add_argument("--dataset", default="dataset", help="Path to dataset root")
    p_enroll.add_argument("--min-samples", type=int, default=5, help="Minimum captures required (5-8)")
    p_enroll.add_argument("--max-samples", type=int, default=8, help="Maximum captures allowed (5-8)")
    p_enroll.add_argument(
        "--allow-update",
        action="store_true",
        help="Allow adding fresh samples for an existing SRN",
    )
    p_enroll.add_argument("--camera", type=int, default=0, help="Laptop camera index")
    p_enroll.add_argument("--capture-width", type=int, default=160, help="Saved face width")
    p_enroll.add_argument("--capture-height", type=int, default=160, help="Saved face height")
    p_enroll.add_argument("--cooldown", type=float, default=0.8, help="Seconds between captures")
    p_enroll.add_argument(
        "--blur-threshold",
        type=float,
        default=80.0,
        help="Minimum blur score (Laplacian variance) for a saved frame",
    )
    p_enroll.add_argument("--model", default="models/eigenclass_model.npz", help="Output model path")
    p_enroll.add_argument(
        "--use-cuda-preprocess",
        action="store_true",
        help="Try CUDA for post-enrollment training preprocessing (falls back to CPU if unavailable)",
    )
    p_enroll.add_argument("--width", type=int, default=64, help="Training resize width")
    p_enroll.add_argument("--height", type=int, default=64, help="Training resize height")
    p_enroll.add_argument("-k", type=int, default=50, help="PCA components")
    p_enroll.set_defaults(func=run_enroll)

    p_rec = subparsers.add_parser("recognize", help="Recognize one face image")
    p_rec.add_argument("--image", required=True, help="Input face image path")
    p_rec.add_argument("--model", default="models/eigenclass_model.npz", help="Path to trained model")
    p_rec.add_argument(
        "--threshold-scale",
        type=float,
        default=1.08,
        help="Multiplier on learned distance threshold (higher = fewer UNKNOWNs)",
    )
    p_rec.add_argument(
        "--margin-threshold",
        type=float,
        default=0.0,
        help="Minimum top1/top2 distance margin for confident match (0 uses model margin)",
    )
    p_rec.add_argument("--mark", action="store_true", help="Write attendance entry")
    p_rec.add_argument("--skip-unknown", action="store_true", help="Do not write UNKNOWN entries")
    p_rec.add_argument(
        "--attendance",
        default="attendance/attendance.csv",
        help="Attendance CSV path",
    )
    p_rec.set_defaults(func=run_recognize)

    p_live = subparsers.add_parser("live", help="Live attendance via laptop camera")
    p_live.add_argument("--model", default="models/eigenclass_model.npz", help="Path to trained model")
    p_live.add_argument("--camera", type=int, default=0, help="Laptop camera index")
    p_live.add_argument("--attendance", default="attendance/attendance.csv", help="Attendance CSV path")
    p_live.add_argument("--confirm-frames", type=int, default=6, help="Frames required before marking PRESENT")
    p_live.add_argument("--min-face", type=int, default=80, help="Minimum detected face size in pixels")
    p_live.add_argument(
        "--track-iou",
        type=float,
        default=0.28,
        help="IoU threshold used to match detections to existing face tracks",
    )
    p_live.add_argument(
        "--track-ttl",
        type=int,
        default=12,
        help="Frames to keep an unseen track alive before dropping it",
    )
    p_live.add_argument(
        "--vote-window",
        type=int,
        default=10,
        help="Per-track rolling vote window for stable identity",
    )
    p_live.add_argument(
        "--min-votes",
        type=int,
        default=4,
        help="Minimum same-SRN votes required before accepting identity",
    )
    p_live.add_argument(
        "--vote-ratio",
        type=float,
        default=0.62,
        help="Required ratio of same-SRN votes in the rolling window",
    )
    p_live.add_argument(
        "--min-track-age",
        type=int,
        default=4,
        help="Minimum frame age before a track can be treated as stable",
    )
    p_live.add_argument(
        "--threshold-scale",
        type=float,
        default=1.10,
        help="Multiplier on learned distance threshold (higher = fewer UNKNOWNs)",
    )
    p_live.add_argument(
        "--margin-threshold",
        type=float,
        default=0.0,
        help="Minimum top1/top2 distance margin for confident match (0 uses model margin)",
    )
    p_live.add_argument(
        "--use-cuda-preprocess",
        action="store_true",
        help="Try CUDA for live face preprocessing (falls back to CPU if unavailable)",
    )
    p_live.add_argument("--mark", dest="mark", action="store_true", help="Enable CSV marking during live mode")
    p_live.add_argument("--no-mark", dest="mark", action="store_false", help="Detection only, do not mark CSV")
    p_live.set_defaults(mark=True, func=run_live)

    p_math = subparsers.add_parser("math", help="Use local Ollama model for mathematics")
    p_math.add_argument("--query", required=True, help="Math question/expression")
    p_math.add_argument("--ollama-model", default="qwen2.5:3b", help="Ollama model name")
    p_math.add_argument("--timeout", type=int, default=240, help="Max wait for Ollama response in seconds")
    p_math.set_defaults(func=run_math)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
