# EIGENCLASS: Automated Attendance with Linear Algebra

This project implements the exact linear algebra flow from your prompt:

1. Convert grayscale face images into vectors and build a data matrix.
2. Compute the mean face.
3. Center the data (subtract mean face).
4. Build covariance matrix.
5. Compute eigenvalues/eigenvectors (Eigenfaces).
6. Reduce dimensionality using PCA (top `k` components).
7. Project faces into face space.
8. Recognize by Euclidean nearest neighbor distance.
9. Mark attendance using a threshold decision.

## Project Structure

```
la_proj/
  context.txt
  eigenclass_attendance.py
  requirements.txt
  README.md
  dataset/
    Student_1/
      img1.jpg
      img2.jpg
    Student_2/
      img1.jpg
  models/
  attendance/
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

## Train

```bash
python eigenclass_attendance.py train --dataset dataset --model models/eigenclass_model.npz --width 64 --height 64 -k 50
```

## Recognize + Mark Attendance

```bash
python eigenclass_attendance.py recognize --image test_face.jpg --model models/eigenclass_model.npz --mark --attendance attendance/attendance.csv
```

## Output

The attendance file is a CSV with:

- `date`
- `time`
- `student`
- `status` (`PRESENT` or `UNKNOWN`)
- `distance`
- `image`

Duplicate `PRESENT` entries for the same student on the same day are prevented.
