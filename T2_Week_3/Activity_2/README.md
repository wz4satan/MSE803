# Week 3 - Activity 2: Data Cleaning Prediction for Missing Values

Continues the data-cleaning work from [Activity 1](../Activity_1/README.md).
For the numeric columns that still have missing values (`Age`, `Net worth`,
`Salary`), this activity predicts the missing cells where a predictor is
available, using **linear regression** and **non-linear (polynomial, degree 2)
regression** — following `nonlinear regression -sample code.py`.

## How to run

```bash
conda activate mse803
pip install -r requirements.txt      # pandas, numpy, matplotlib, tabulate, scikit-learn
python predict_missing.py
```

Requires the Activity 1 cleaned dataset at `../Activity_1/results/cleaned_data.csv`.

## Outputs (`results/`)

| File | Purpose |
|---|---|
| `prediction_report.md` | Full comparison report (methods, fit metrics, predictions, conclusion). |
| `predicted_data.csv` | Cleaned data with missing cells filled (linear predictions) + `Estimated` column. |
| `charts/` | 5 PNG charts — linear & polynomial fits with predicted points per task. |

## Approach

- **Retrieve:** the two duplicate `ID=2` (Bob) records are complementary
  (one has `Salary`, the other `Age`+`Net worth`), so merging them **retrieves**
  Bob's missing values directly — used as ground truth to cross-check.
- **Predict (regression):** for each `(target, predictor)` pair, train on
  complete-case rows, then predict missing cells. Two models compared:
  - Linear regression (degree 1)
  - Polynomial regression (degree 2, `PolynomialFeatures`)

## Key results

| Task | n | Linear R² | Polynomial R² |
|---|---|---|---|
| Age ~ Salary | 7 | 0.398 | 0.395 |
| Net worth ~ Salary | 6 | 0.020 | 0.018 |
| Net worth ~ Age | 7 | 0.235 | 0.251 |
| Salary ~ Age | 7 | 0.398 | 0.419 |
| Salary ~ Net worth | 6 | 0.020 | 0.428 |

Predicted values (linear / polynomial):

| Cell | Predictor | Linear | Polynomial | True (retrieved) |
|---|---|---|---|---|
| Bob Age | Salary=60,000 | 28.7 | 28.6 | 30 |
| Bob Net worth | Salary=60,000 | 38,403 | 38,376 | 35,000 |
| Bob Salary | Age=30 | 62,527 | 61,455 | 60,000 |
| Bob Salary | Net worth=35,000 | 61,936 | 66,438 | 60,000 |
| David Net worth | Age=38 | 47,885 | 48,555 | — |

## Conclusion (which method is better)

- **Polynomial fits the training data better in-sample** (avg R² 0.302 vs 0.214),
  but with only 6–7 points and 3 parameters it is prone to **overfitting**
  (e.g. `Salary ~ Net worth` jumps from R² 0.02 → 0.43).
- **Cross-check on Bob** gives mixed results: linear is closer for Age and
  Salary-from-Net worth; polynomial is closer for Salary-from-Age and
  Net worth (marginally). No clear winner.
- Because the relationships are weak (many R² ≈ 0.02–0.42), **neither model is
  reliable here**. For a small sample we prefer the **linear** model for
  stability, and the **retrieved** duplicate value (Bob) is the most reliable.
- `predicted_data.csv` therefore fills the missing cells with the **linear**
  predictions, and marks every estimated cell in the `Estimated` column.
- David's Net worth has no ground truth and the models give a wide range
  (≈ 41,000–49,000), so it should be treated as a low-confidence estimate.
