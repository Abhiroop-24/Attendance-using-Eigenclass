# EIGENCLASS Quick README

## Made by:
Abhishek Kumar Singh (PES2UG24CS026).
Abhiroop (PES2UG24CS024).

## What this project is

This is an automated attendance system based on face recognition using linear algebra and PCA (Eigenfaces).
It takes student face images, learns a compact face space, and then identifies a new face to mark attendance.

## What was built

- Core script: `eigenclass_attendance.py`
- Dependencies list: `requirements.txt`
- Project brief/context: `context.txt`

The script supports:

- Training a model from folder-based student images
- Recognizing a new image as known student or UNKNOWN
- Writing attendance to CSV with timestamp
- Avoiding duplicate PRESENT entries for the same student on the same date

## Linear algebra pipeline used

1. Convert images to grayscale vectors and form a matrix.
2. Compute mean face.
3. Center all vectors by subtracting mean.
4. Compute covariance matrix.
5. Perform eigen decomposition to get Eigenfaces.
6. Keep top-k components (PCA reduction).
7. Project training and test faces into reduced subspace.
8. Match with Euclidean nearest neighbor.
9. Apply threshold decision boundary for known vs UNKNOWN.

## Expected folder layout

```
la_proj/
  context.txt
  eigenclass_attendance.py
  requirements.txt
  README.md
  dataset/
    Alice/
      img1.jpg
      img2.jpg
    Bob/
      img1.jpg
  models/
  attendance/
```

## Setup

```bash
py -3 -m pip install -r requirements.txt
```

## Train model

```bash
py -3 eigenclass_attendance.py train --dataset dataset --model models/eigenclass_model.npz --width 64 --height 64 -k 50
```

## Recognize and mark attendance

```bash
py -3 eigenclass_attendance.py recognize --image test_face.jpg --model models/eigenclass_model.npz --mark --attendance attendance/attendance.csv
```

## Attendance CSV fields

- date
- time
- student
- status (PRESENT or UNKNOWN)
- distance
- image

## Notes

- If `dataset` does not exist, training will fail.
- Recognition quality depends heavily on real face data quality and threshold tuning.
