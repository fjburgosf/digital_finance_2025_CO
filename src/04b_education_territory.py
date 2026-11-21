"""Step 04b: territorial gaps in online banking within each educational level.

For the adults of each educational level (primary or less, secondary, technical or higher; current students are
classified by the level in which they are enrolled), the gap in online banking of Cesar, of the rest of the
Caribbean and of the whole Caribbean with respect to the rest of Colombia is decomposed with the same exact
decomposition, bootstrap (Rao-Wu, 999 replicates, seed 20261120) and disclosure rules as step 04. The replicate
multipliers are drawn for the whole adult sample, as in step 04, so the estimates of the educational levels share
one set of replicates.

Outputs: results/03_education/decomposition_by_education.csv, prevalences_by_education.csv, meta.json

Usage (from the repository root):  python src/04b_education_territory.py
"""
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "03_education"
_spec = importlib.util.spec_from_file_location("dec04", Path(__file__).with_name("04_decomposition.py"))
dec04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dec04)
warnings.filterwarnings("ignore", category=RuntimeWarning)
LEVELS = ["Primary or less", "Secondary", "Technical or higher"]
GROUPINGS = {"territory": ("Rest of Colombia", ["Cesar", "Rest of the Caribbean"]),
             "caribbean": ("Rest of Colombia", ["Caribbean"])}


def q(x, p):
    return float(np.nanquantile(x, p))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(dec04.PROC / "analytic_15plus.parquet")
    d = d[d.age >= 18].copy()
    d["wU"], d["wB"] = d["w"] * d["U"], d["w"] * d["B"]
    rng = np.random.default_rng(dec04.SEED)
    psu_stratum = d.groupby("psu")["stratum"].first()
    psu_index, mult = dec04.replicate_multipliers(psu_stratum, rng)
    prev_rows, dec_rows = [], []
    for lev in LEVELS:
        ds = d[d.education == lev]
        for var, (ref, others) in GROUPINGS.items():
            est = {g: dec04.estimates(s, mult) for g, s in dec04.group_sums(ds, var, psu_index).items()}
            for g, (pt, reps, n, _) in est.items():
                for k, nm in enumerate(["P_U", "P_B_given_U", "P_B"]):
                    se = np.nanstd(reps[:, k], ddof=1)
                    prev_rows.append({"education": lev, "grouping": var, "group": g, "measure": nm, "n": n,
                                      "estimate": pt[k], "lo": q(reps[:, k], 0.025), "hi": q(reps[:, k], 0.975),
                                      "cv": se / pt[k] if pt[k] else np.nan, "suppressed": n < 30})
            ptr, repr_, nr, _ = est[ref]
            for g in others:
                ptg, repg, ng, _ = est[g]
                tU, tC = np.log(ptg[0] / ptr[0]), np.log(ptg[1] / ptr[1])
                tot = tU + tC
                rU, rC = np.log(repg[:, 0] / repr_[:, 0]), np.log(repg[:, 1] / repr_[:, 1])
                rT = rU + rC
                bad = ~np.isfinite(rT)
                rU, rC, rT = (np.where(bad, np.nan, x) for x in (rU, rC, rT))
                with np.errstate(divide="ignore", invalid="ignore"):
                    share = rU / rT
                big = abs(tot) >= 0.05
                dec_rows.append({"education": lev, "grouping": var, "group": g, "reference": ref, "n_group": ng,
                                 "n_ref": nr, "ratio_B": np.exp(tot), "ratio_B_lo": np.exp(q(rT, 0.025)),
                                 "ratio_B_hi": np.exp(q(rT, 0.975)), "term_internet": tU,
                                 "term_internet_lo": q(rU, 0.025), "term_internet_hi": q(rU, 0.975),
                                 "term_conditional": tC, "term_conditional_lo": q(rC, 0.025),
                                 "term_conditional_hi": q(rC, 0.975), "total_log_gap": tot,
                                 "share_internet": tU / tot if big else np.nan,
                                 "share_internet_lo": q(share, 0.025) if big else np.nan,
                                 "share_internet_hi": q(share, 0.975) if big else np.nan,
                                 "invalid_replicates": int(bad.sum()), "suppressed": min(ng, nr) < 30})
    prev, dec = pd.DataFrame(prev_rows), pd.DataFrame(dec_rows)
    prev.round(5).to_csv(OUT / "prevalences_by_education.csv", index=False)
    dec.round(5).to_csv(OUT / "decomposition_by_education.csv", index=False)
    meta = {"n": int(len(d)), "replicates": dec04.R, "seed": dec04.SEED, "n_psu": int(len(psu_index)),
            "n_by_education": d.education.value_counts().to_dict()}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    print(dec[["education", "group", "n_group", "ratio_B", "ratio_B_lo", "ratio_B_hi", "term_internet",
               "term_conditional", "share_internet"]].round(3).to_string())
    pv = prev[(prev.measure != "P_B") & prev.group.isin(["Cesar", "Rest of Colombia", "Caribbean"])]
    print(pv.drop_duplicates(["education", "group", "measure"])[["education", "group", "measure", "estimate", "lo", "hi"]].round(3).to_string())


if __name__ == "__main__":
    main()
