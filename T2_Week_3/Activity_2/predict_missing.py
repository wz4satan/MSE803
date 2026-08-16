"""
Week 3 - Activity 2: Data cleaning prediction for missing values
================================================================

Continues the data-cleaning work from Week 3 Activity 1. For numeric
columns that still have missing values (`Age`, `Net worth`, `Salary`), this
script predicts the missing cells where a predictor is available, using:

  1. Linear regression (degree 1)
  2. Non-linear / polynomial regression (degree 2, following the sample code)

For each (target, predictor) pair it:
  - trains on the complete-case rows,
  - evaluates both models (MSE, RMSE, R2),
  - predicts the missing values with both models,
  - cross-checks Bob's predictions against the value "retrieved" by merging
    the two duplicate ID=2 records (complementary fields),
  - saves charts and a full comparison report.

Run:
    python predict_missing.py

Outputs (in this folder):
    results/prediction_report.md   comparison report
    results/predicted_data.csv     cleaned data + estimated cells
    results/charts/*.png           regression fits with predicted points
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "results")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
for d in (RESULTS_DIR, CHARTS_DIR):
    os.makedirs(d, exist_ok=True)

NUMERIC = ["Age", "Net worth", "Salary"]

# --------------------------------------------------------------------------- #
# 1. Load the cleaned dataset from Week 3 Activity 1
# --------------------------------------------------------------------------- #
CLEANED = os.path.join(BASE, "..", "Activity_1", "results", "cleaned_data.csv")
df = pd.read_csv(CLEANED)
df["ID"] = df["ID"].astype("Int64")
df["Name"] = df["Name"].astype("string")
df["Country"] = df["Country"].astype("string")
for c in NUMERIC:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# --- Retrieve complementary values from the duplicate ID=2 records ---------- #
# Row "2,Bob" has Salary=60000; the other "2,Bob" has Age=30 & Net worth=35000.
# Merging the two records *retrieves* Bob's missing values (no model needed).
bob_true = {"Age": 30.0, "Net worth": 35000.0, "Salary": 60000.0}

# --------------------------------------------------------------------------- #
# 2. Define the regression tasks (target <- predictor)
# --------------------------------------------------------------------------- #
TASKS = [
    ("Age", "Salary"),  # Bob (Salary=60000) has Age missing
    ("Net worth", "Salary"),  # Bob (Salary=60000), David (Salary=68000)
    ("Net worth", "Age"),  # David (Age=38)
    ("Salary", "Age"),  # Bob (Age=30) has Salary missing
    ("Salary", "Net worth"),  # Bob (Net worth=35000)
]


def fit_and_predict(target: str, predictor: str) -> dict:
    """Fit linear & polynomial (deg 2) models, return metrics + predictions."""
    train = df[[target, predictor]].dropna()
    X = train[predictor].to_numpy(dtype=float).reshape(-1, 1)
    y = train[target].to_numpy(dtype=float)

    # rows to predict: target missing but predictor present
    mask = df[target].isna() & df[predictor].notna()
    pred_rows = df.index[mask].tolist()
    pred_x = df.loc[mask, predictor].to_numpy(dtype=float).reshape(-1, 1)

    # --- linear (degree 1) ---
    lin = LinearRegression().fit(X, y)
    y_lin = lin.predict(X)
    lin_pred = lin.predict(pred_x).ravel()

    # --- polynomial (degree 2) ---
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xp = poly.fit_transform(X)
    pol = LinearRegression().fit(Xp, y)
    y_pol = pol.predict(Xp)
    pol_pred = pol.predict(poly.transform(pred_x)).ravel()

    return {
        "target": target,
        "predictor": predictor,
        "n_train": len(X),
        "rows": pred_rows,
        "pred_x": pred_x.ravel(),
        "lin_pred": lin_pred,
        "pol_pred": pol_pred,
        "lin": {
            "mse": mean_squared_error(y, y_lin),
            "rmse": float(np.sqrt(mean_squared_error(y, y_lin))),
            "r2": r2_score(y, y_lin),
            "coef": lin.coef_[0],
            "intercept": lin.intercept_,
        },
        "pol": {
            "mse": mean_squared_error(y, y_pol),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pol))),
            "r2": r2_score(y, y_pol),
            "coefs": pol.coef_,
            "intercept": pol.intercept_,
        },
        "X": X.ravel(),
        "y": y,
    }


tasks = [fit_and_predict(t, p) for t, p in TASKS]


# --------------------------------------------------------------------------- #
# 3. Charts per task
# --------------------------------------------------------------------------- #
def make_charts() -> list[str]:
    files = []
    for res in tasks:
        tgt, pred = res["target"], res["predictor"]
        X, y = res["X"], res["y"]
        xs = np.linspace(X.min(), X.max(), 100)
        # linear line
        lin_y = res["lin"]["intercept"] + res["lin"]["coef"] * xs
        # polynomial curve
        pf = PolynomialFeatures(degree=2, include_bias=False)
        pol_y = (
            LinearRegression()
            .fit(pf.fit_transform(X.reshape(-1, 1)), y)
            .predict(pf.transform(xs.reshape(-1, 1)))
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X, y, color="#4C72B0", s=45, label="training (complete cases)")
        ax.plot(xs, lin_y, color="#55A868", lw=2, label="linear fit (deg 1)")
        ax.plot(
            xs, pol_y, color="#C44E52", lw=2, ls="--", label="polynomial fit (deg 2)"
        )
        for i, (px, lp, pp) in enumerate(
            zip(res["pred_x"], res["lin_pred"], res["pol_pred"])
        ):
            ax.scatter(
                [px],
                [lp],
                color="#55A868",
                marker="^",
                s=90,
                zorder=5,
                label="linear prediction" if i == 0 else None,
            )
            ax.scatter(
                [px],
                [pp],
                color="#C44E52",
                marker="s",
                s=90,
                zorder=5,
                label="polynomial prediction" if i == 0 else None,
            )
            ax.annotate(
                f"lin={lp:,.0f}\npoly={pp:,.0f}",
                (px, max(lp, pp)),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
            )
        ax.set_xlabel(pred)
        ax.set_ylabel(tgt)
        ax.set_title(f"Predict {tgt} from {pred} (n_train={res['n_train']})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(
            CHARTS_DIR,
            f"{tgt.lower().replace(' ', '_')}_from_{pred.lower().replace(' ', '_')}.png",
        )
        fig.savefig(p, dpi=150)
        plt.close(fig)
        files.append(p)
    return files


# --------------------------------------------------------------------------- #
# 4. Build the report
# --------------------------------------------------------------------------- #
def build_report(charts: list[str]) -> str:
    md = []
    md.append("# Week 3 Activity 2 — Missing Value Prediction")
    md.append("")
    md.append("Data source: cleaned dataset from Activity 1 (10 records).")
    md.append("")
    md.append("## 1. Missing values to predict")
    md.append("")
    md.append("| Column | Missing rows | Predictable? |")
    md.append("|---|---|---|")
    md.append(
        "| Age | Bob (ID=2), Heidi (ID=9) | Bob: yes (has Salary); Heidi: no (no predictor) |"
    )
    md.append(
        "| Net worth | Bob (ID=2), David (ID=5), Heidi (ID=9) | Bob/David: yes; Heidi: no |"
    )
    md.append("| Salary | Bob (ID=2), Heidi (ID=9) | Bob: yes; Heidi: no |")
    md.append("")
    md.append("Heidi has **no** numeric predictor values, so her cells cannot be")
    md.append("estimated by regression and stay missing.\n")
    md.append("For Bob, the two duplicate ID=2 records are complementary (one has")
    md.append("`Salary=60000`, the other `Age=30, Net worth=35000`), so merging them")
    md.append("**retrieves** his missing values directly — these act as ground truth")
    md.append("to cross-check the regression predictions.\n")

    md.append("## 2. Method")
    md.append("")
    md.append("For each (target, predictor) pair, the model is trained on the rows")
    md.append("where **both** values are present, then used to predict the rows where")
    md.append("the target is missing. Two models are compared (per the sample code):")
    md.append("- **Linear regression** — `LinearRegression` (degree 1).")
    md.append(
        "- **Polynomial regression** — `LinearRegression` on `PolynomialFeatures(degree=2)`.\n"
    )

    md.append("## 3. Model fit per task")
    md.append("")
    md.append("| Task | n_train | Model | MSE | RMSE | R² |")
    md.append("|---|---|---|---|---|---|")
    for res in tasks:
        for model, tag in (("lin", "Linear"), ("pol", "Polynomial deg 2")):
            m = res[model]
            md.append(
                f"| {res['target']} ~ {res['predictor']} | {res['n_train']} | {tag} "
                f"| {m['mse']:,.2f} | {m['rmse']:,.2f} | {m['r2']:.4f} |"
            )
    md.append("")

    md.append("## 4. Predicted missing values (linear vs polynomial)")
    md.append("")
    md.append(
        "| Target | Row | Predictor value | Linear prediction | Polynomial prediction | Retrieved / true value |"
    )
    md.append("|---|---|---:|---:|---:|---:|")
    row_label = {1: "Bob (ID=2)", 2: "Bob (ID=2)", 4: "David (ID=5)", 8: "Heidi (ID=9)"}
    truth_map = {
        (1, "Age"): bob_true["Age"],
        (1, "Net worth"): bob_true["Net worth"],
        (2, "Salary"): bob_true["Salary"],
    }
    for res in tasks:
        for i, r in enumerate(res["rows"]):
            label = row_label.get(r, f"row {r + 2}")
            truth = truth_map.get((r, res["target"]))
            truth_s = f"{truth:,.0f}" if truth is not None else "—"
            md.append(
                f"| {res['target']} | {label} | {res['pred_x'][i]:,.0f} "
                f"| {res['lin_pred'][i]:,.2f} | {res['pol_pred'][i]:,.2f} | {truth_s} |"
            )
    md.append("")

    md.append(
        "> **Note:** the final filled dataset (`predicted_data.csv`) uses the "
        "**linear** predictions (recommended for stability on this small sample); "
        "the polynomial predictions are shown here for comparison.\n"
    )

    md.append("## 5. Visualisations")
    md.append("")
    for ch in charts:
        cap = os.path.splitext(os.path.basename(ch))[0].replace("_", " ").title()
        md.append(f"### {cap}\n")
        md.append(f"![{cap}](charts/{os.path.basename(ch)})\n")

    md.append("## 6. Comparison & conclusion")
    md.append("")
    # summary of R2 by model
    lin_r2 = [r["lin"]["r2"] for r in tasks]
    pol_r2 = [r["pol"]["r2"] for r in tasks]
    avg_lin, avg_pol = float(np.mean(lin_r2)), float(np.mean(pol_r2))
    md.append(
        f"- Average in-sample R²: **linear {avg_lin:.4f}** vs "
        f"**polynomial {avg_pol:.4f}**."
    )
    md.append(
        f"- Polynomial fits the training data {('better' if avg_pol >= avg_lin else 'worse')} "
        f"in-sample, but with only {min(t['n_train'] for t in tasks)}–"
        f"{max(t['n_train'] for t in tasks)} training points a degree-2 model (3 parameters) "
        f"is prone to **overfitting**."
    )
    # Bob cross-check (which model is closer to the retrieved true value)
    md.append(
        "- **Cross-check on Bob** (comparing each model's prediction to the retrieved "
        "true value):"
    )
    bob_rows = {
        (1, "Age", "Salary"): bob_true["Age"],
        (1, "Net worth", "Salary"): bob_true["Net worth"],
        (2, "Salary", "Age"): bob_true["Salary"],
        (2, "Salary", "Net worth"): bob_true["Salary"],
    }
    for res in tasks:
        for i, r in enumerate(res["rows"]):
            key = (r, res["target"], res["predictor"])
            if key not in bob_rows:
                continue
            true = bob_rows[key]
            el = abs(res["lin_pred"][i] - true)
            ep = abs(res["pol_pred"][i] - true)
            winner = "linear" if el < ep else ("polynomial" if ep < el else "tie")
            md.append(
                f"  - {res['target']} (true {true:,.0f}) from {res['predictor']}: "
                f"linear error {el:,.1f} vs polynomial error {ep:,.1f} → **{winner}** closer."
            )
    md.append("")
    md.append(
        "- **David's Net worth** has no ground truth; the two predictors give a wide "
        "range (≈ 41,000–49,000), so treat it as a rough estimate with **low confidence**."
    )
    md.append(
        "- **Recommendation:** with only 6–7 training points and weak relationships "
        "(many R² ≈ 0.02–0.42), neither model is clearly better. Prefer the **linear** model "
        "for stability; when available, use the **retrieved** value from duplicate records "
        "(Bob) as the most reliable. Use polynomial predictions only if they agree with the "
        "linear ones."
    )
    md.append("")
    return "\n".join(md)


# --------------------------------------------------------------------------- #
# 5. Filled dataset
# --------------------------------------------------------------------------- #
def best_filled() -> pd.DataFrame:
    """Fill missing cells using the LINEAR predictions (the recommended, stable
    method on this small sample). When a cell can be predicted by more than one
    predictor, the predictor with the highest linear R2 is used."""
    filled = df.copy()
    used = {}
    # collect candidates per (row, target): (predictor, linear R2, prediction)
    cands: dict = {}
    for res in tasks:
        for i, r in enumerate(res["rows"]):
            key = (r, res["target"])
            cands.setdefault(key, []).append(
                (res["predictor"], res["lin"]["r2"], res["lin_pred"][i])
            )
    for (r, tgt), cand in cands.items():
        pred, _r2, val = max(cand, key=lambda x: x[1])
        filled.at[r, tgt] = val
        used[(r, tgt)] = f"{pred} (linear)"
    # build an Estimated flag per row listing which cells were estimated
    est = []
    for i in range(len(filled)):
        cells = [f"{col}={used[(i, col)]}" for col in NUMERIC if (i, col) in used]
        est.append("; ".join(cells) if cells else "")
    filled["Estimated"] = est
    return filled


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    charts = make_charts()
    report = build_report(charts)
    report_path = os.path.join(RESULTS_DIR, "prediction_report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    filled = best_filled()
    csv_path = os.path.join(RESULTS_DIR, "predicted_data.csv")
    out = filled.copy()
    for col in NUMERIC:
        out[col] = [_fmt_val(x) for x in out[col]]
    out.to_csv(csv_path, index=False)

    print("=" * 72)
    print("WEEK 3 ACTIVITY 2 - MISSING VALUE PREDICTION")
    print("=" * 72)
    print("\n[Model fit (MSE / RMSE / R2)]")
    for res in tasks:
        for model, tag in (("lin", "Linear"), ("pol", "Poly2")):
            m = res[model]
            print(
                f"  {res['target']:<10} ~ {res['predictor']:<10} {tag:<8} "
                f"MSE={m['mse']:>12,.2f}  RMSE={m['rmse']:>9,.2f}  R2={m['r2']:.4f}"
            )
    print("\n[Predictions]")
    for res in tasks:
        for i, r in enumerate(res["rows"]):
            print(
                f"  row{r + 2:>2} {res['target']:<10} <- {res['predictor']:<10} "
                f"(x={res['pred_x'][i]:,.0f})  linear={res['lin_pred'][i]:,.2f}  "
                f"poly={res['pol_pred'][i]:,.2f}"
            )
    print(f"\nReport written to: {report_path}")
    print(f"Predicted dataset: {csv_path}")
    print(f"Charts: {CHARTS_DIR}")


def _fmt_val(x):
    return "" if pd.isna(x) else f"{float(x):,.0f}".replace(",", "")


if __name__ == "__main__":
    main()
