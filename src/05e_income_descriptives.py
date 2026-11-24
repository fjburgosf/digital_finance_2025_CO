"""Step 05e: internet use and online banking by quintile of per capita household income and territory (adults).

Weighted prevalences with the expansion factors (descriptive support for the sensitivity analysis with income of
step 05); cells with fewer than 30 unweighted persons are not reported.

Output: results/02_two_part/income/prevalences_by_income.csv, income_distribution.csv

Usage (from the repository root):  python src/05e_income_descriptives.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = pd.read_parquet(ROOT / "data" / "processed" / "analytic_15plus.parquet")
    d = d[d.age >= 18]
    rows = []
    for t in ("Cesar", "Rest of the Caribbean", "Rest of Colombia"):
        for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
            g = d[(d.territory == t) & (d.income_q == q)]
            u = g[g.U == 1]
            row = {"territory": t, "income_quintile": q, "n": len(g), "n_users": len(u),
                   "P_U": np.average(g.U, weights=g.w) if len(g) >= 30 else np.nan,
                   "P_B_given_U": np.average(u.B, weights=u.w) if len(u) >= 30 else np.nan,
                   "P_B": np.average(g.B, weights=g.w) if len(g) >= 30 else np.nan}
            rows.append(row)
    out = pd.DataFrame(rows)
    path = ROOT / "results" / "02_two_part" / "income" / "prevalences_by_income.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    dist = (d.groupby(["territory", "income_q"])["w"].sum() / d.groupby("territory")["w"].sum()).rename("share").reset_index()
    dist.to_csv(path.parent / "income_distribution.csv", index=False)
    print(out.round(3).to_string())
    print(dist.round(3).to_string())


if __name__ == "__main__":
    main()
