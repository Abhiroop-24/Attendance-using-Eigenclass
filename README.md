# EIGENCLASS Smart Attendance

## Made by
- Abhishek Kumar Singh - PES2UG24CS026
- Abhiroop - PES2UG24CS024
- Aayaan - PES2UG24CS017
- Adarsh - PES2UG24CS028

## What this version does

This upgraded version keeps the same linear algebra Eigenfaces pipeline, and adds a full laptop-camera attendance flow:

- Webcam registration with automatic face detection
- Capture exactly 5 images per student by default
- Store student Name + SRN (for example: CS024)
- Retrain model after registration (optional one-shot)
- Live webcam attendance mode with real-time detection and marking
- Duplicate protection: one PRESENT per SRN per day
- Ollama math assistant mode using `qwen2.5:3b`

## Linear algebra pipeline used

1. Convert images to grayscale vectors and form matrix $X$.
2. Compute mean face $\mu$.
3. Center vectors: $A = X - \mu$.
4. Compute covariance matrix: $C = \frac{A^T A}{n-1}$.
5. Eigen decomposition on $C$.
6. Keep top-$k$ eigenvectors (Eigenfaces).
7. Project faces into PCA space.
8. Nearest-neighbor match using Euclidean distance.
9. Threshold decision for KNOWN vs UNKNOWN.

## Folder layout

```
LA_pog/
  eigenclass_attendance.py
  requirements.txt
  README.md
  dataset/
    student_registry.csv
    CS024/
      cs024_01.jpg
      cs024_02.jpg
      ...
  models/
    eigenclass_model.npz
  attendance/
    attendance.csv
```

## Setup

```bash
python3 -m pip install -r requirements.txt
```

If Linux blocks global pip installs (PEP 668), use a local virtual environment:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Final setup flow (recommended)

1) Register student from laptop camera (captures 5 photos and retrains immediately):

```bash
python3 eigenclass_attendance.py register \
  --name "Abhiroop" \
  --srn CS024 \
  --samples 5 \
  --train-after
```

2) Start live attendance mode:

```bash
python3 eigenclass_attendance.py live --mark
```

When a person comes in front of the laptop camera, the system detects the face and marks attendance after confirmation frames.

One-command final setup (prompts Name + SRN, captures photos, retrains, starts live mode):

```bash
python3 launch_attendance.py
```

## All commands

Train model:

```bash
python3 eigenclass_attendance.py train --dataset dataset --model models/eigenclass_model.npz --width 64 --height 64 -k 50
```

Alternative training script:

```bash
python3 train_ai.py --dataset dataset --model models/eigenclass_model.npz --width 64 --height 64 -k 50
```

Register (webcam + face detection + Name + SRN):

```bash
python3 eigenclass_attendance.py register --name "Student Name" --srn CS024 --samples 5
```

Recognize one image:

```bash
python3 eigenclass_attendance.py recognize --image sample_inputs/alice_test.jpg --mark
```

Live attendance via laptop camera:

```bash
python3 eigenclass_attendance.py live --mark --confirm-frames 6
```

Math assistant using local Ollama `qwen2.5:3b`:

```bash
python3 eigenclass_attendance.py math --query "Solve 2x + 7 = 31"
```

## Attendance CSV fields

- date
- time
- srn
- name
- status (PRESENT or UNKNOWN)
- distance
- source (image path or camera index)

## Notes

- Webcam features require OpenCV (`opencv-python`).
- For better accuracy, keep face well-lit and centered during registration.
- If your old model file only had `labels`, it is still supported.
