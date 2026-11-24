"""Step 04c: main reason for not using the internet among adult non-users (ECV 2025, P1929), by territory.

Codes as printed in the ECV 2025 questionnaire (chapter on information and communication technologies, question 7):
1 too expensive; 2 not necessary; 3 does not know how to use it; 4 no coverage; 5 not allowed; 6 security or
privacy; 7 age; 8 health; 9 other. Weighted shares with the expansion factors; the denominator of each territory is
every adult non-user (all of them answered the question), reported as the unweighted number of persons and the
weighted population.

Output: results/01_decomposition/reasons_nonuse.csv

Usage (from the repository root):  python src/04c_reasons_nonuse.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["Too expensive", "Not necessary", "Does not know how to use it", "No coverage", "Not allowed",
         "Security or privacy", "Age", "Health", "Other"]


def main():
    d = pd.read_parquet(ROOT / "data" / "processed" / "analytic_15plus.parquet")
    nu = d[(d.age >= 18) & (d.U == 0)]
    assert nu.reason_nonuse.notna().all()
    scopes = {"Colombia": nu, "Caribbean": nu[nu.territory != "Rest of Colombia"], "Cesar": nu[nu.territory == "Cesar"],
              "Rest of the Caribbean": nu[nu.territory == "Rest of the Caribbean"],
              "Rest of Colombia": nu[nu.territory == "Rest of Colombia"]}
    rows = []
    for scope, g in scopes.items():
        W = g.w.sum()
        for k in ORDER:
            s = g[g.reason_nonuse == k]
            rows.append({"scope": scope, "reason": k, "n": len(s), "share": s.w.sum() / W,
                         "denominator_n": len(g), "denominator_weighted": round(W)})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "01_decomposition" / "reasons_nonuse.csv", index=False)
    print(out.pivot(index="reason", columns="scope", values="share").reindex(ORDER).round(3).to_string())
    print(out.groupby("scope")[["denominator_n", "denominator_weighted"]].first())


if __name__ == "__main__":
    main()
