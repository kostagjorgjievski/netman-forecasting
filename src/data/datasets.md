# datasets.py — Sliding Window Dataset for Multivariate Time Series

This module provides a clean, reproducible, and model-agnostic data pipeline for multivariate time-series forecasting.  
It standardizes data preprocessing across all models (DeepAR, iTransformer, PatchTST) by enforcing a **consistent format** and **stable scaling procedure**.

---

## Overview

**Purpose:**  
Convert a multivariate CSV file into fixed-length input–output windows suitable for PyTorch models.  
The **last column is always treated as the prediction target**.

**Core flow:**
1. Load the CSV file.
2. Clean NaN and ±inf values.
3. Scale features using the first 70% of time (to prevent data leakage).
4. Build valid sliding windows `(x, y)`.
5. Split windows into train/val/test subsets.
6. Create PyTorch `DataLoader` objects for each split.

---

## Input and Output

| Symbol | Meaning | Shape |
|:-------|:---------|:------|
| `T` | number of time steps | |
| `D` | number of features (columns) | |
| `L` | sequence length (`seq_len`) | |
| `H` | prediction length (`pred_len`) | |

Each sample consists of:
- `x`: past context `[L, D]`
- `x_mark`: optional time features `[L, 0]` (currently placeholder)
- `y`: future target `[H, 1]`

---

## Functions

### `_clean_matrix(arr: np.ndarray) -> np.ndarray`

Cleans invalid numeric values column-wise.

**Steps:**
1. Replace ±inf with NaN.
2. Forward-fill and backward-fill missing values for temporal continuity.
3. Replace any remaining NaN with 0.0.

**Why:**  
Models cannot handle NaN or infinite values.  
Forward/backward filling preserves realistic continuity.  
Zero replacement is used only for leading or trailing gaps.

**Complexity:**  
- Time: O(T × D)  
- Space: O(T × D)

---

### `_fit_scale(train: np.ndarray) -> (mu, sd)`

Computes column-wise mean and standard deviation on the **training-time region** (first 70% of rows).  
Small standard deviations are clamped to `1.0` to avoid division by zero.

**Purpose:**  
Standardize input data for stable optimization and consistent scaling across datasets.

---

### `_apply_scale(x, mu, sd) -> np.ndarray`

Applies z-score normalization `(x - mu) / sd`.

**Why:**  
Decouples fitting from applying.  
Allows the same `(mu, sd)` to be reused for validation, test, and inverse-scaling predictions.

---

## Class `SlidingWindowDataset`

Implements the PyTorch `Dataset` abstraction for time-series windows.

### Initialization

```python
SlidingWindowDataset(data, seq_len, pred_len)
