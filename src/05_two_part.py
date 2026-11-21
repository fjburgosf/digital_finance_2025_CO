"""Step 05: adjusted two-part model and territorial counterfactuals, and the Fairlie-type aggregate decomposition
of the territorial gap (sensitivity).

Two weighted logit models for adults (18+), with indicators for age group, sex, education, area, ethnicity and
territory (Cesar, rest of the Caribbean, rest of Colombia; reference = rest of Colombia):
  part 1: U (internet use), all adults;     part 2: B | U (online banking), internet users.
For the adults of each territory T (Cesar; rest of the Caribbean; the whole Caribbean), with predicted
probabilities pU and pC:
  observed (model)      P(B)_T      = mean[pU(T) * pC(T)]
  CF-internet           P(B)_T^U    = mean[pU(rest) * pC(T)]   (national internet use, own conditional use)
  CF-conditional        P(B)_T^C    = mean[pU(T) * pC(rest)]   (own internet use, national conditional use)
  CF-both               P(B)_T^UC   = mean[pU(rest) * pC(rest)]
so that ln[P(B)_T / P(B)_T^UC] is the covariate-adjusted territorial gap, split into the internet and conditional
stages (each evaluated with the other stage at the territory's own level and at the national level; their mean is
reported). Average marginal effects of territory on U, on B|U and on B are also reported.

Fairlie-type aggregate decomposition (pooled logit for B among all adults without territory indicators, coefficients
of the rest of Colombia): explained (composition) = P(X_rest, b_rest) - P(X_T, b_rest); unexplained = remainder.

Uncertainty: the same PSU bootstrap as step 04 (Rao-Wu, 999 replicates, seed 20261120), refitting every model.

Outputs: results/02_two_part/<tag>/counterfactuals.csv, coefficients.csv, fairlie.csv

Usage (from the repository root):  python src/05_two_part.py --tag main [--reps 999]
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
spec = importlib.util.spec_from_file_location("dec", ROOT / "src" / "04_decomposition.py")
dec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dec)
COVS = {"age_group": "18-29", "sex": "Men", "education": "Technical or higher", "area": "Urban (cabecera)",
        "ethnicity": "None"}
TERR = ["Cesar", "Rest of the Caribbean"]


def design(d, with_territory=True):
    cols = []
    for v, ref in COVS.items():
        for lev in sorted(d[v].unique()):
            if lev != ref:
                cols.append(pd.Series((d[v] == lev).astype(float).to_numpy(), name=f"{v}={lev}"))
    X = pd.concat(cols, axis=1)
    if with_territory:
        for t in TERR:
            X[f"territory={t}"] = (d["territory"] == t).astype(float).to_numpy()
    return X


def fit(X, y, w):
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500, tol=1e-8)
    m.fit(X.to_numpy(), y, sample_weight=w)
    return m


def set_territory(X, t):
    Z = X.copy()
    for c in [c for c in Z.columns if c.startswith("territory=")]:
        Z[c] = 1.0 if c == f"territory={t}" else 0.0
    return Z


def wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def one_fit(d, X, w):
    users = d["U"].to_numpy() == 1
    mU = fit(X, d["U"].to_numpy(), w)
    mC = fit(X[users], d["B"].to_numpy()[users], w[users])
    res = {}
    for t, mask in (("Cesar", d.territory == "Cesar"), ("Rest of the Caribbean", d.territory == "Rest of the Caribbean"),
                    ("Caribbean", d.territory != "Rest of Colombia")):
        mask = mask.to_numpy()
        Xt, wt = X[mask], w[mask]
        Xr = set_territory(Xt, "Rest of Colombia")
        pU_t, pU_r = mU.predict_proba(Xt.to_numpy())[:, 1], mU.predict_proba(Xr.to_numpy())[:, 1]
        pC_t, pC_r = mC.predict_proba(Xt.to_numpy())[:, 1], mC.predict_proba(Xr.to_numpy())[:, 1]
        obs, cfU, cfC, cfUC = (wmean(a * b, wt) for a, b in ((pU_t, pC_t), (pU_r, pC_t), (pU_t, pC_r), (pU_r, pC_r)))
        total = np.log(obs / cfUC)
        termU = 0.5 * (np.log(obs / cfU) + np.log(cfC / cfUC))
        termC = 0.5 * (np.log(obs / cfC) + np.log(cfU / cfUC))
        res[t] = {"P_B_model": obs, "P_B_cf_internet": cfU, "P_B_cf_conditional": cfC, "P_B_cf_both": cfUC,
                  "adj_ratio_B": float(np.exp(total)), "adj_term_internet": float(termU),
                  "adj_term_conditional": float(termC), "adj_share_internet": float(termU / total) if abs(total) >= 0.05 else np.nan,
                  "ame_U": wmean(pU_t - pU_r, wt), "ame_C": wmean(pC_t - pC_r, wt), "ame_B": obs - cfUC}
    coefs = {"U": dict(zip(X.columns, mU.coef_[0])), "C": dict(zip(X.columns, mC.coef_[0]))}
    return res, coefs


def fairlie(d, w):
    X = design(d, with_territory=False)
    rest = (d.territory == "Rest of Colombia").to_numpy()
    m = fit(X[rest], d["B"].to_numpy()[rest], w[rest])
    out = {}
    for t, mask in (("Cesar", d.territory == "Cesar"), ("Caribbean", d.territory != "Rest of Colombia")):
        mask = mask.to_numpy()
        p_rest = wmean(m.predict_proba(X[rest].to_numpy())[:, 1], w[rest])
        p_t_brest = wmean(m.predict_proba(X[mask].to_numpy())[:, 1], w[mask])
        obs_rest, obs_t = wmean(d["B"].to_numpy()[rest], w[rest]), wmean(d["B"].to_numpy()[mask], w[mask])
        gap = obs_rest - obs_t
        expl = p_rest - p_t_brest
        out[t] = {"gap_rest_minus_T": gap, "explained_composition": expl, "unexplained": gap - expl,
                  "share_explained": expl / gap if abs(gap) > 0.005 else np.nan}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--reps", type=int, default=999)
    a = ap.parse_args()
    t0 = time.time()
    out = ROOT / "results" / "02_two_part" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(PROC / "analytic_15plus.parquet")
    d = d[d.age >= 18].reset_index(drop=True)
    X = design(d)
    w = d["w"].to_numpy()
    w = w / w.mean()
    point, coefs = one_fit(d, X, w)
    fl = fairlie(d, w)
    rng = np.random.default_rng(dec.SEED)
    psu_stratum = d.groupby("psu")["stratum"].first()
    psu_index, mult = dec.replicate_multipliers(psu_stratum, rng)
    pos = pd.Series(np.arange(len(psu_index)), index=psu_index)[d["psu"]].to_numpy()
    reps, freps = [], []
    for r in range(a.reps):
        wr = w * mult[r, pos]
        keep = wr > 0
        dr, Xr = d[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
        res, _ = one_fit(dr, Xr, wr[keep])
        reps.append(res)
        freps.append(fairlie(dr, wr[keep]))
        if (r + 1) % 100 == 0:
            print(f"replicate {r + 1}/{a.reps} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    rows = []
    for t, est in point.items():
        for k, v in est.items():
            vals = np.array([rp[t][k] for rp in reps], dtype=float)
            rows.append({"territory": t, "quantity": k, "estimate": v, "lo": np.nanquantile(vals, 0.025),
                         "hi": np.nanquantile(vals, 0.975)})
    pd.DataFrame(rows).to_csv(out / "counterfactuals.csv", index=False)
    frows = []
    for t, est in fl.items():
        for k, v in est.items():
            vals = np.array([rp[t][k] for rp in freps], dtype=float)
            frows.append({"territory": t, "quantity": k, "estimate": v, "lo": np.nanquantile(vals, 0.025),
                          "hi": np.nanquantile(vals, 0.975)})
    pd.DataFrame(frows).to_csv(out / "fairlie.csv", index=False)
    pd.DataFrame(coefs).to_csv(out / "coefficients.csv")
    meta = {"tag": a.tag, "n": int(len(d)), "n_users": int(d.U.sum()), "reps": a.reps, "seed": dec.SEED,
            "runtime_min": round((time.time() - t0) / 60, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    print(pd.DataFrame(rows).round(4).to_string())
    print(pd.DataFrame(frows).round(4).to_string())


if __name__ == "__main__":
    main()
