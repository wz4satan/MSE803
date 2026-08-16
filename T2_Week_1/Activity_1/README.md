# Week 1 - Activity 2: IRIS Dataset Exploration and Analysis

## Overview

This project explores the classic **Iris flower dataset** from the UCI Machine Learning Repository and answers two key questions:

- How many features and classes are available?
- Are there any duplicate records in the dataset?

## Understanding

The Iris dataset is one of the most well-known benchmark datasets in machine learning. It contains **150 samples** of iris flowers, each described by four physical measurements. Every sample belongs to one of three iris species (classes).

The four features are all **continuous** measurements (in centimetres):

| Feature | Description |
|---------|-------------|
| `sepal length` | Length of the sepal (cm) |
| `sepal width`  | Width of the sepal (cm) |
| `petal length` | Length of the petal (cm) |
| `petal width`  | Width of the petal (cm) |

The target variable is **categorical** — it labels each sample as one of three iris species:

- `Iris-setosa`
- `Iris-versicolor`
- `Iris-virginica`

## Findings

Running the analysis script (`analysis.py`) produced the following results:

| Question | Answer |
|----------|--------|
| Number of records | 150 |
| Number of features | 4 |
| Number of classes | 3 |
| Class names | `Iris-setosa`, `Iris-versicolor`, `Iris-virginica` |
| Number of duplicate records | 3 |

**Records per class** — the dataset is perfectly balanced, with exactly 50 samples per class:

| Class | Count |
|-------|-------|
| Iris-setosa | 50 |
| Iris-versicolor | 50 |
| Iris-virginica | 50 |

### Key observations

1. **4 features, 3 classes**: The dataset uses four continuous measurements to distinguish three species, which makes it a great example for classification tasks.
2. **Balanced classes**: Each class contributes exactly 50 records (50/50/50), so there is no class-imbalance issue to handle.
3. **3 duplicate records**: Duplicated rows were detected using `DataFrame.duplicated()`. In practice, duplicate samples are usually removed or reviewed before training a model.

## Steps Followed

1. **Fetch the dataset** — downloaded the Iris dataset from the UCI Machine Learning Repository using the `ucimlrepo` package (`fetch_ucirepo(id=53)`).
2. **Separate features and target** — split the data into the feature matrix `X` and the target vector `y`.
3. **Combine into one dataframe** — concatenated features and target into a single `pandas` `DataFrame` for convenient analysis.
4. **Compute dataset statistics**:
   - Number of features via `X.shape[1]`.
   - Number of classes via `y["class"].nunique()`.
   - Class names via `y["class"].unique()`.
   - Records per class via `y["class"].value_counts()`.
   - Duplicate records via `df.duplicated().sum()`.
5. **Print and interpret the results**.

## How to Run

### Prerequisites

- Python 3.8+
- `pandas`
- `ucimlrepo`

Install dependencies:

```bash
pip install pandas ucimlrepo
```

### Run the script

```bash
python analysis.py
```

Expected output:

```text
Number of records: 150
Number of features: 4
Number of classes: 3
Class names: ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
Number of records per class:
class
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
Number of duplicate records: 3
```

## Repository Structure

```
Activity_1/
├── analysis.py   # Main analysis script
└── README.md     # This file
```

## Reference

- UCI Machine Learning Repository — Iris dataset: https://archive.ics.uci.edu/ml/datasets/iris
