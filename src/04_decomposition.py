"""Step 04: weighted prevalences and exact decomposition of the online-banking gap.

For each group g and its reference r:
    ln[P(B)_g / P(B)_r] = ln[P(U)_g / P(U)_r] + ln[P(B|U)_g / P(B|U)_r]
with P(U) = internet use, P(B|U) = online banking among users and P(B) = P(U) P(B|U). The share of each stage is
its term divided by the total (reported only when |total| >= 0.05, i.e. a gap of about 5%).

Uncertainty: bootstrap of primary sampling units within strata (Rao-Wu rescaling, n_h - 1 PSUs drawn with
replacement in each stratum; strata with one PSU keep multiplier 1), 999 replicates, seed 20261120. Percentile 95%
intervals. Disclosure control: cells with fewer than 30 unweighted persons are suppressed; estimates with a
coefficient of variation above 15% are flagged (DANE recommendation).

Scopes: Colombia (all groupings), the Caribbean region and Cesar (groupings within the region).
Options: --population 18 (main) or 15 (sensitivity); --weekly (U = at least weekly use) [sensitivity];
--caribbean-no-sai (the Caribbean without San Andrés) [sensitivity].

Outputs: results/01_decomposition/<tag>/prevalences.csv, decomposition.csv, reasons_nonuse.csv

Usage (from the repository root):  python src/04_decomposition.py --tag main
"""
import argparse
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SEED, R = 20261120, 999
warnings.filterwarnings("ignore", category=RuntimeWarning)  # empty replicate cells give NaN; counted in output
GROUPINGS = {
    "territory": ("Rest of Colombia", ["Cesar", "Rest of the Caribbean"]),
    "caribbean": ("Rest of Colombia", ["Caribbean"]),
    "area": ("Urban (cabecera)", ["Rural (centros poblados y rural disperso)"]),
    "sex": ("Men", ["Women"]),
    "age_group": ("18-29", ["30-44", "45-59", "60+"]),
    "education": ("Technical or higher", ["Secondary", "Primary or less"]),
    "ethnicity": ("None", ["Indigenous", "Black or Afro-Colombian"]),
}
WITHIN = ["area", "sex", "age_group", "education", "ethnicity"]


def replicate_multipliers(psu_stratum, rng):
    """Rao-Wu rescaling bootstrap multipliers, shape (R, n_psu)."""
    psus = psu_stratum.index.to_numpy()
    strata = psu_stratum.to_numpy()
    mult = np.ones((R, len(psus)))
    for h in np.unique(strata):
        idx = np.where(strata == h)[0]
        n = len(idx)
        if n < 2:
            continue
        draws = rng.integers(0, n, size=(R, n - 1))
        counts = np.zeros((R, n))
        for k in range(n - 1):
            np.add.at(counts, (np.arange(R), draws[:, k]), 1)
        mult[:, idx] = counts * n / (n - 1)
    return pd.Index(psus, name="psu"), mult


def group_sums(d, var, psu_index):
    """PSU x level sums of w, wU, wB (and unweighted n)."""
    g = d.groupby([var, "psu"]).agg(w=("w", "sum"), wU=("wU", "sum"), wB=("wB", "sum"), n=("w", "size")).reset_index()
    out = {}
    for lev, sub in g.groupby(var):
        s = sub.drop(columns=var).set_index("psu").reindex(psu_index, fill_value=0)
        out[lev] = s
    return out


def estimates(s, mult):
    """Point and replicate estimates of P(U), P(B|U), P(B) from PSU sums."""
    W, WU, WB = (s[c].to_numpy() for c in ("w", "wU", "wB"))
    point = np.array([WU.sum() / W.sum(), WB.sum() / WU.sum(), WB.sum() / W.sum()])
    rw, ru, rb = mult @ W, mult @ WU, mult @ WB
    reps = np.column_stack([ru / rw, rb / ru, rb / rw])
    return point, reps, int(s["n"].sum()), int((s["n"] * (s["wU"] > 0)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--population", type=int, default=18)
    ap.add_argument("--weekly", action="store_true")
    ap.add_argument("--caribbean-no-sai", action="store_true")
    a = ap.parse_args()
    out = ROOT / "results" / "01_decomposition" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(PROC / "analytic_15plus.parquet")
    d = d[d.age >= a.population].copy()
    if a.weekly:
        d["U"], d["B"] = d["U_weekly"], d["B_weekly"]
    if a.caribbean_no_sai:
        d = d[d["dept"] != "88"].copy()
    d["wU"], d["wB"] = d["w"] * d["U"], d["w"] * d["B"]
    rng = np.random.default_rng(SEED)
    psu_stratum = d.groupby("psu")["stratum"].first()
    psu_index, mult = replicate_multipliers(psu_stratum, rng)
    d = d.assign(all="All")
    scopes = {"Colombia": d, "Caribbean": d[d["caribbean"] == "Caribbean"], "Cesar": d[d["territory"] == "Cesar"]}
    prev_rows, dec_rows = [], []
    for scope, ds in scopes.items():
        groupings = dict(GROUPINGS) if scope == "Colombia" else {k: GROUPINGS[k] for k in WITHIN}
        groupings["all"] = ("All", [])
        for var, (ref, others) in groupings.items():
            sums = group_sums(ds, var, psu_index)
            est = {lev: estimates(s, mult) for lev, s in sums.items()}
            for lev, (pt, reps, n, nu) in est.items():
                for k, nm in enumerate(["P_U", "P_B_given_U", "P_B"]):
                    se = np.nanstd(reps[:, k], ddof=1)
                    prev_rows.append({"scope": scope, "grouping": var, "group": lev, "measure": nm, "n": n,
                                      "invalid_replicates": int(np.isnan(reps[:, k]).sum()),
                                      "estimate": pt[k], "lo": np.nanquantile(reps[:, k], 0.025),
                                      "hi": np.nanquantile(reps[:, k], 0.975), "cv": se / pt[k] if pt[k] else np.nan,
                                      "suppressed": n < 30, "cv_gt_15": (se / pt[k] > 0.15) if pt[k] else True})
            if var == "all" or ref not in est:
                continue
            ptr, repr_, nr, _ = est[ref]
            for lev in others:
                if lev not in est:
                    continue
                ptg, repg, ng, _ = est[lev]
                tU = np.log(ptg[0] / ptr[0])
                tC = np.log(ptg[1] / ptr[1])
                tot = tU + tC
                rU = np.log(repg[:, 0] / repr_[:, 0])
                rC = np.log(repg[:, 1] / repr_[:, 1])
                rT = rU + rC
                bad = ~np.isfinite(rT)
                rU, rC, rT = (np.where(bad, np.nan, x) for x in (rU, rC, rT))
                with np.errstate(divide="ignore", invalid="ignore"):
                    share = rU / rT
                row = {"scope": scope, "grouping": var, "group": lev, "reference": ref, "n_group": ng, "n_ref": nr,
                       "ratio_B": np.exp(tot), "ratio_B_lo": np.exp(np.nanquantile(rT, 0.025)),
                       "ratio_B_hi": np.exp(np.nanquantile(rT, 0.975)),
                       "term_internet": tU, "term_internet_lo": np.nanquantile(rU, 0.025),
                       "term_internet_hi": np.nanquantile(rU, 0.975),
                       "term_conditional": tC, "term_conditional_lo": np.nanquantile(rC, 0.025),
                       "term_conditional_hi": np.nanquantile(rC, 0.975), "total_log_gap": tot,
                       "share_internet": tU / tot if abs(tot) >= 0.05 else np.nan,
                       "share_internet_lo": np.nanquantile(share, 0.025) if abs(tot) >= 0.05 else np.nan,
                       "share_internet_hi": np.nanquantile(share, 0.975) if abs(tot) >= 0.05 else np.nan,
                       "invalid_replicates": int(np.isnan(rT).sum() + np.isinf(rT).sum()),
                       "suppressed": min(ng, nr) < 30}
                dec_rows.append(row)
        # reasons for not using the internet among non-users (weighted shares), by territory scope
        nu = ds[(ds.U == 0) & ds.reason_nonuse.notna()]
        r = nu.groupby("reason_nonuse")["w"].sum() / nu["w"].sum()
        for k, val in r.items():
            prev_rows.append({"scope": scope, "grouping": "reason_nonuse", "group": k, "measure": "share_of_nonusers",
                              "n": int((nu.reason_nonuse == k).sum()), "estimate": val})
    prev = pd.DataFrame(prev_rows)
    dec = pd.DataFrame(dec_rows)
    prev.round(5).to_csv(out / "prevalences.csv", index=False)
    dec.round(5).to_csv(out / "decomposition.csv", index=False)
    meta = {"tag": a.tag, "population": a.population, "weekly": a.weekly, "caribbean_no_sai": a.caribbean_no_sai,
            "n": int(len(d)), "replicates": R, "seed": SEED, "n_psu": int(len(psu_index)),
            "strata_single_psu": int((psu_stratum.value_counts() < 2).sum())}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    show = dec[["scope", "grouping", "group", "ratio_B", "term_internet", "term_conditional", "share_internet",
                "share_internet_lo", "share_internet_hi"]]
    print(show.round(3).to_string())


if __name__ == "__main__":
    main()
