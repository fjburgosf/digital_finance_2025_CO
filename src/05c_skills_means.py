"""Step 05c: weighted mean number of digital skills (P1710S1-S12, 0-12) among adult internet users, by territory
and educational level (descriptive support for the sensitivity analysis with digital skills of step 05).

Output: results/02_two_part/skills/skills_means.csv

Usage (from the repository root):  python src/05c_skills_means.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = pd.read_parquet(ROOT / "data" / "processed" / "analytic_15plus.parquet")
    d = d[(d.age >= 18) & (d.U == 1) & d.skills.notna()]
    rows = []
    for (t, e), g in pd.concat([d, d.assign(education="All")]).groupby(["territory", "education"]):
        rows.append({"territory": t, "education": e, "n": len(g), "mean_skills": np.average(g.skills, weights=g.w)})
    out = pd.DataFrame(rows)
    path = ROOT / "results" / "02_two_part" / "skills" / "skills_means.csv"
    out.round(4).to_csv(path, index=False)
    print(out.round(2).to_string())


if __name__ == "__main__":
    main()
