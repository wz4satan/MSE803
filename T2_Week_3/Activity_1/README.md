# Basic Data Analytics — Sample Dataset (MSE803, T2 Week 3, Activity 1)

A Python program that imports and cleans `Sample_dataset.csv`, computes the four
lecture metrics (**Mean / Variance / Std Dev / Covariance**) for both the
Population and the Sample (per the lecture screenshot formulas), visualises the
results, and writes a full report.

- **Program:** [`analysis.py`](analysis.py)
- **Raw data:** [`Sample_dataset.csv`](Sample_dataset.csv)
- **Latest result:** [`results/report.md`](results/report.md)

---

## How to run

Requires Python 3.10+ and the packages in [`requirements.txt`](requirements.txt)
(`pandas`, `numpy`, `matplotlib`, `tabulate`).

```bash
# 1. Create / activate a Python environment (e.g. conda)
conda create -n mse803 python=3.12 -y
conda activate mse803

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python analysis.py
```

Outputs written to `results/`:

```
results/
├── report.md               # import → cleaning → metrics → charts
└── charts/                 # 3 PNG charts (data quality, metrics, covariance)
```

---

## The metrics in this analysis

Four metrics are computed for each numeric column (`Age`, `Net worth`,
`Salary`), for both the **Population** (N) and the **Sample** (n), following the
lecture screenshot formulas.

### 1. Mean (μ / x̄)

- Formula: μ = Σx_i / N  ·  x̄ = Σx_i / n
- **What it measures:** the average (central/typical) value of the data.
- **How to interpret:** the "balance point" of the data; it is pulled toward any
  extreme (outlier) values.

### 2. Variance (σ² / S²)

- Formula: σ² = Σ(x_i − μ)² / N  ·  S² = Σ(x_i − x̄)² / n
- **What it measures:** the average squared deviation from the mean — the overall
  spread of the data.
- **How to interpret:** larger value → data is more spread out; but the units are
  squared, so it is hard to read directly.

### 3. Standard Deviation (σ / S)

- Formula: σ = √σ²  ·  S = √S²
- **What it measures:** the typical distance of a value from the mean, back in the
  original units.
- **How to interpret:** the most intuitive measure of spread — e.g. for a roughly
  normal distribution, ~68% of values lie within ±1 standard deviation of the mean.

### 4. Covariance (σ_xy / S_xy)

- Formula: σ_xy = Σ(x_i − μx)(y_i − μy) / N  ·  S_xy = Σ(x_i − x̄)(y_i − ȳ) / (n−1)
- **What it measures:** how two variables move together (linear association).
- **How to interpret:** positive → they tend to increase together; negative → one
  rises while the other falls; near 0 → no linear relationship. (The magnitude
  depends on the units, so compare the sign, not the size.)

> **Note:** in the lecture screenshot the sample variance S² also divides by n,
> so S² equals σ² numerically. The usual *unbiased* estimator divides by (n−1)
> (Bessel's correction). The sample covariance divides by (n−1).

---

## Key results (summary)

### Data cleaning
- Quoted comma (`"30,000"` → 30000), word-numbers (`thirty-eight` → 38,
  `sixty five thousand` → 65000), country code (`AU` → `AUS`) and invalid dates
  are handled before computing.
- Missing values are excluded per column: Age n=8, Net worth n=7, Salary n=8.
  Covariance uses only the rows where **both** variables are present.

### Mean / Variance / Std Dev

| Variable | N | Mean (μ/x̄) | Variance σ² | Std Dev σ | Sample Var S² | Sample SD S |
|---|---:|---:|---:|---:|---:|---:|
| Age | 8 | 30.75 | 35.44 | 5.95 | 35.44 | 5.95 |
| Net worth | 7 | 38,571.43 | 171,959,183.67 | 13,113.32 | 171,959,183.67 | 13,113.32 |
| Salary | 8 | 62,625.00 | 27,984,375.00 | 5,290.03 | 27,984,375.00 | 5,290.03 |

### Covariance

| Pair | n | σ_xy (/N) | S_xy (/(n−1)) |
|---|---:|---:|---:|
| Age ↔ Net worth | 7 | 35,877.55 | 41,857.14 |
| Age ↔ Salary | 7 | 22,285.71 | 26,000.00 |
| Net worth ↔ Salary | 6 | 10,972,222.22 | 13,166,666.67 |

All three covariances are **positive**, so each pair of variables tends to move
in the same direction (Age ↔ Salary weakest, Age ↔ Net worth strongest of the three).

---

## Files

| File | Purpose |
|---|---|
| `analysis.py` | Computes Mean / Variance / Std Dev / Covariance (Population vs Sample) and writes the latest result. |
| `requirements.txt` | Python dependencies for reproducible setup. |
| `Sample_dataset.csv` | The raw input data. |
| `results/report.md` | Latest result — full report (data import, cleaning, metrics, charts). |
| `results/charts/` | Charts: data quality, metrics, covariance scatter. |
