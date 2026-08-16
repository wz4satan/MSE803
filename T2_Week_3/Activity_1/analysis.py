"""
Data Analysis — Sample Dataset (MSE803, T2 Week 3, Activity 1)
==============================================================

End-to-end analysis of `Sample_dataset.csv`:

  1. Import the raw data and inspect it (shape, columns, missing values)
  2. Clean the data (thousands separators, word-numbers, country codes, dates)
  3. Compute the four lecture metrics — Mean / Variance / Std Dev / Covariance
     (Population vs Sample, following the lecture screenshot formulas)
  4. Visualise the results (charts)
  5. Write a Markdown report (`results/report.md`) that shows the key result
     of every step and embeds the charts

Run:
    python analysis.py
"""

from __future__ import annotations

import os
import re

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (safe for scripts/CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "results")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
for d in (RESULTS_DIR, CHARTS_DIR):
    os.makedirs(d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Data cleaning helpers
# --------------------------------------------------------------------------- #
_WORD_NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
    "million": 1_000_000,
}


def words_to_number(text: str) -> float | None:
    """Convert English number words (e.g. 'sixty five thousand') to a number."""
    if text is None:
        return None
    tokens = re.split(r"\s+", str(text).strip().lower().replace("-", " "))
    if not tokens or any(t not in _WORD_NUM for t in tokens):
        return None
    total, current = 0.0, 0.0
    for token in tokens:
        val = _WORD_NUM[token]
        if val >= 1000:
            current = (current or 1) * val
            total += current
            current = 0.0
        elif val == 100:
            current = (current or 1) * val
        else:
            current += val
    return total + current


def to_number(series: pd.Series) -> pd.Series:
    """Best-effort numeric conversion: strip thousands separators, try pandas,
    then fall back to word-number parsing. Non-numeric results become NaN."""

    def _prep(x):
        if not isinstance(x, str):
            return x
        s = x.strip().replace(",", "")  # remove thousands separators
        return s if s else np.nan

    s = series.map(_prep)
    numeric = pd.to_numeric(s, errors="coerce")
    mask = numeric.isna() & s.notna()
    parsed = s[mask].map(words_to_number)
    numeric.loc[mask] = pd.to_numeric(parsed, errors="coerce")
    return numeric


# --------------------------------------------------------------------------- #
# 1. Data import
# --------------------------------------------------------------------------- #
raw = pd.read_csv(os.path.join(BASE, "Sample_dataset.csv"))
raw.columns = [c.strip() for c in raw.columns]

# --------------------------------------------------------------------------- #
# 2. Data cleaning
# --------------------------------------------------------------------------- #
data = pd.DataFrame(
    {
        "Age": to_number(raw["Age"]),
        "Net worth": to_number(raw["Net worth"]),
        "Salary": to_number(raw["Salary"]),
    }
)
country_clean = pd.Series(
    [
        (
            np.nan
            if pd.isna(v)
            else {"AU": "AUS"}.get(str(v).strip().upper(), str(v).strip().upper())
        )
        for v in raw["Country"]
    ],
    index=raw.index,
)
date_clean = pd.to_datetime(raw["Join Date"], dayfirst=True, errors="coerce")

# Full cleaned dataset (shown in the report)
cleaned_df = pd.DataFrame(
    {
        "ID": raw["ID"],
        "Name": raw["Name"],
        "Age": data["Age"],
        "Net worth": data["Net worth"],
        "Country": country_clean,
        "Salary": data["Salary"],
        "Join Date": date_clean,
    }
)

VARS = ["Age", "Net worth", "Salary"]


def var_and_std(values, denom):
    """Variance = Σ(x−x̄)² / denom  and its square root."""
    mean = values.mean()
    var = ((values - mean) ** 2).sum() / denom
    return var, float(np.sqrt(var))


def _fmt(x, digits=2):
    return f"{x:.{digits}f}"


def _fmt_int(x):
    """Format a number with thousands separators; blank when missing."""
    return "" if pd.isna(x) else f"{x:,.0f}"


def _fmt_date(x):
    return "" if pd.isna(x) else x.strftime("%Y-%m-%d")


def _fmt_str(x):
    return "" if pd.isna(x) else str(x)


def _fmt_csv_num(x):
    """Number for CSV output: integer when whole, blank when missing."""
    if pd.isna(x):
        return ""
    return f"{float(x):,.0f}".replace(",", "")


# --------------------------------------------------------------------------- #
# 3. Compute the four metrics
# --------------------------------------------------------------------------- #
results = []
for col in VARS:
    v = data[col].dropna().to_numpy(dtype=float)
    n = len(v)
    mean = v.mean()
    var_pop, sd_pop = var_and_std(v, n)  # σ² / N
    var_samp, sd_samp = var_and_std(v, n)  # S² / n  (as in screenshot)
    results.append((col, n, mean, var_pop, sd_pop, var_samp, sd_samp))

cov_pairs = []
for i, a in enumerate(VARS):
    for b in VARS[i + 1 :]:
        pair = data[[a, b]].dropna().to_numpy(dtype=float)
        n = len(pair)
        x, y = pair[:, 0], pair[:, 1]
        xm, ym = x.mean(), y.mean()
        cross = ((x - xm) * (y - ym)).sum()
        cov_pop = cross / n  # σ_xy
        cov_samp = cross / (n - 1) if n > 1 else np.nan  # S_xy
        cov_pairs.append((a, b, n, x, y, cov_pop, cov_samp))


# --------------------------------------------------------------------------- #
# 4. Charts
# --------------------------------------------------------------------------- #
def make_charts() -> list[str]:
    files = []

    # 4.1 Missing values per column (data quality)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    miss = raw.isna().sum().sort_values()
    colors = ["#C44E52" if m > 0 else "#55A868" for m in miss.values]
    miss.plot(kind="bar", ax=ax, color=colors)
    ax.set_title("Missing values per column (before cleaning)")
    ax.set_ylabel("Count")
    for i, v in enumerate(miss.values):
        ax.text(i, v + 0.05, str(v), ha="center", va="bottom")
    fig.tight_layout()
    p = os.path.join(CHARTS_DIR, "data_quality.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    files.append(p)

    # 4.2 Mean / Variance / Std Dev
    names = [r[0] for r in results]
    means = [r[2] for r in results]
    varss = [r[3] for r in results]
    sds = [r[4] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].bar(names, means, color="#4C72B0")
    axes[0].set_title("Mean (μ/x̄)")
    for i, v in enumerate(means):
        axes[0].text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    axes[1].bar(names, varss, color="#DD8452")
    axes[1].set_title("Variance (σ², log scale)")
    axes[1].set_yscale("log")
    for i, v in enumerate(varss):
        axes[1].text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    axes[2].bar(names, sds, color="#55A868")
    axes[2].set_title("Std Dev (σ)")
    for i, v in enumerate(sds):
        axes[2].text(i, v, f"{v:,.2f}", ha="center", va="bottom", fontsize=8)
    for ax in axes:
        ax.set_ylabel("value")
    fig.suptitle("Calculated metrics per variable (Population)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(CHARTS_DIR, "metrics.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    files.append(p)

    # 4.3 Covariance scatter plots
    fig, axes = plt.subplots(1, len(cov_pairs), figsize=(15, 4.5))
    if len(cov_pairs) == 1:
        axes = [axes]
    for ax, (a, b, n, x, y, cov_pop, _) in zip(axes, cov_pairs):
        ax.scatter(x, y, color="#4C72B0", s=45)
        if len(x) > 1 and x.std() > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slope * xs + intercept, color="#C44E52", lw=2, label="trend")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title(f"Cov(σ_xy) = {cov_pop:,.0f}   (n={n})")
        ax.legend()
    fig.suptitle(
        "Covariance — pairwise scatter (all positive → move together)", fontsize=13
    )
    fig.tight_layout()
    p = os.path.join(CHARTS_DIR, "covariance_scatter.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    files.append(p)

    return files


# --------------------------------------------------------------------------- #
# 5. Build the report
# --------------------------------------------------------------------------- #
def cleaning_log() -> list[tuple]:
    log = []
    for col in VARS:
        for rv, cv in zip(raw[col], data[col]):
            if pd.isna(rv):
                continue
            rs = str(rv).strip()
            if pd.isna(cv):
                log.append((col, rs, "(missing)", "unparseable → treated as missing"))
            elif "," in rs:
                log.append((col, rs, f"{cv:,.0f}", "removed thousands separator"))
            elif any(ch.isalpha() for ch in rs):
                log.append((col, rs, f"{cv:,.0f}", "word number → numeric"))
    for rv, cv in zip(raw["Country"], country_clean):
        if pd.isna(rv):
            continue
        rs = str(rv).strip()
        if rs.upper() != str(cv):
            log.append(("Country", rs, str(cv), "standardised country code"))
    for rv, cv in zip(raw["Join Date"], date_clean):
        if pd.isna(rv):
            continue
        if pd.isna(cv):
            log.append(
                ("Join Date", str(rv).strip(), "(missing)", "invalid date dropped")
            )
    return log


def build_report(charts: list[str]) -> str:
    md = []
    md.append("# Data Analysis Report — Sample Dataset")
    md.append("")
    md.append("Generated by `analysis.py` from `Sample_dataset.csv`.")
    md.append("")

    # --- 1. Import -------------------------------------------------------- #
    md.append("## 1. Data Import")
    md.append("")
    md.append(f"- Records loaded: **{len(raw)}** rows × **{raw.shape[1]}** columns.")
    md.append(f"- Columns: `{', '.join(raw.columns)}`")
    md.append("")
    md.append("Raw data preview (first 5 rows):")
    md.append("")
    md.append(raw.head(5).fillna("").to_markdown(index=False))
    md.append("")

    # --- 2. Cleaning ------------------------------------------------------ #
    md.append("## 2. Data Cleaning")
    md.append("")
    md.append("### 2.1 Missing values (before cleaning)")
    md.append("")
    miss = raw.isna().sum()
    miss_df = pd.DataFrame(
        {
            "Column": miss.index,
            "Missing": miss.values,
            "Missing %": (miss.values / len(raw) * 100).round(1),
        }
    )
    md.append(miss_df.to_markdown(index=False))
    md.append("")
    md.append(f"![Data quality](charts/{os.path.basename(charts[0])})")
    md.append("")
    md.append("### 2.2 Cleaning transformations applied")
    md.append("")
    log = cleaning_log()
    if log:
        log_df = pd.DataFrame(log, columns=["Column", "Raw value", "Cleaned", "Action"])
        md.append(log_df.to_markdown(index=False))
    md.append("")
    md.append("### 2.3 Effective sample sizes after cleaning")
    md.append("")
    md.append("| Variable | n |")
    md.append("|---|---:|")
    for col, n, *_ in results:
        md.append(f"| {col} | {n} |")
    md.append("")
    md.append("### 2.4 Cleaned dataset (after cleaning)")
    md.append("")
    md.append("| ID | Name | Age | Net worth | Country | Salary | Join Date |")
    md.append("|---:|---|---:|---:|---|---:|---|")
    for i in range(len(cleaned_df)):
        md.append(
            f"| {_fmt_int(cleaned_df['ID'].iloc[i])} "
            f"| {_fmt_str(cleaned_df['Name'].iloc[i])} "
            f"| {_fmt_int(cleaned_df['Age'].iloc[i])} "
            f"| {_fmt_int(cleaned_df['Net worth'].iloc[i])} "
            f"| {_fmt_str(cleaned_df['Country'].iloc[i])} "
            f"| {_fmt_int(cleaned_df['Salary'].iloc[i])} "
            f"| {_fmt_date(cleaned_df['Join Date'].iloc[i])} |"
        )
    md.append("")

    # --- 3. Metrics ------------------------------------------------------- #
    md.append("## 3. Calculated Metrics")
    md.append("")
    md.append("### 3.1 Mean / Variance / Std Dev")
    md.append("")
    md.append(
        "| Variable | N | Mean (μ/x̄) | Variance σ² (/N) | Std Dev σ | Sample Var S² (/n) | Sample SD S |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for col, n, mean, var_pop, sd_pop, var_samp, sd_samp in results:
        md.append(
            f"| {col} | {n} | {_fmt(mean)} | {_fmt(var_pop)} | {_fmt(sd_pop, 4)} "
            f"| {_fmt(var_samp)} | {_fmt(sd_samp, 4)} |"
        )
    md.append("")
    md.append(f"![Metrics](charts/{os.path.basename(charts[1])})")
    md.append("")
    md.append("### 3.2 Covariance")
    md.append("")
    md.append("| Pair | n | Covariance σ_xy (/N) | Covariance S_xy (/(n−1)) |")
    md.append("|---|---:|---:|---:|")
    for a, b, n, *_rest, cov_pop, cov_samp in cov_pairs:
        md.append(f"| {a} ↔ {b} | {n} | {_fmt(cov_pop)} | {_fmt(cov_samp)} |")
    md.append("")
    md.append(f"![Covariance scatter](charts/{os.path.basename(charts[2])})")
    md.append("")

    # --- 4. Notes --------------------------------------------------------- #
    md.append("## 4. Notes")
    md.append("")
    md.append("- Population (N) = the whole column; Sample (n) = the observed values.")
    md.append(
        "- As written in the lecture screenshot, the sample variance S² divides by n"
    )
    md.append(
        "  (same as σ²); the usual unbiased estimator divides by (n−1) (Bessel's correction)."
    )
    md.append("- Sample covariance S_xy divides by (n−1).")
    md.append(
        "- Single-variable stats use that column's non-missing values; covariance uses"
    )
    md.append("  only the rows where BOTH variables are present.")
    md.append("")
    return "\n".join(md)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    charts = make_charts()
    report = build_report(charts)
    report_path = os.path.join(RESULTS_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    # Export the cleaned dataset as CSV (dates formatted, blanks for missing)
    csv_df = pd.DataFrame(
        {
            "ID": [_fmt_csv_num(x) for x in cleaned_df["ID"]],
            "Name": [_fmt_str(x) for x in cleaned_df["Name"]],
            "Age": [_fmt_csv_num(x) for x in cleaned_df["Age"]],
            "Net worth": [_fmt_csv_num(x) for x in cleaned_df["Net worth"]],
            "Country": [_fmt_str(x) for x in cleaned_df["Country"]],
            "Salary": [_fmt_csv_num(x) for x in cleaned_df["Salary"]],
            "Join Date": [_fmt_date(x) for x in cleaned_df["Join Date"]],
        }
    )
    csv_path = os.path.join(RESULTS_DIR, "cleaned_data.csv")
    csv_df.to_csv(csv_path, index=False)

    print("=" * 78)
    print("DATA ANALYSIS — Sample Dataset")
    print("=" * 78)
    print("\n[A] Data import")
    print(f"    {len(raw)} rows x {raw.shape[1]} columns")
    print("\n[B] Data cleaning (missing values before cleaning)")
    print("    " + raw.isna().sum().to_string().replace("\n", "\n    "))
    print("\n[C] Mean / Variance / Std Dev")
    print(
        f"{'Variable':<10} {'N':>3} {'Mean':>10} {'Var σ²':>12} {'SD σ':>9} {'Var S²':>12} {'SD S':>9}"
    )
    print("-" * 78)
    for col, n, mean, var_pop, sd_pop, var_samp, sd_samp in results:
        print(
            f"{col:<10} {n:>3} {mean:>10.2f} {var_pop:>12.2f} "
            f"{sd_pop:>9.4f} {var_samp:>12.2f} {sd_samp:>9.4f}"
        )
    print("\n[D] Covariance")
    print(f"{'Pair':<22} {'n':>3} {'Cov σ_xy (/N)':>15} {'Cov S_xy (/(n−1))':>18}")
    print("-" * 78)
    for a, b, n, *_rest, cov_pop, cov_samp in cov_pairs:
        print(f"{a + ' ↔ ' + b:<22} {n:>3} {cov_pop:>15.2f} {cov_samp:>18.2f}")
    print(f"\nReport written to: {report_path}")
    print(f"Charts written to: {CHARTS_DIR}")
    print(f"Cleaned dataset written to: {csv_path}")


if __name__ == "__main__":
    main()
