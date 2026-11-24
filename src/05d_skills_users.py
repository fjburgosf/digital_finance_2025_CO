"""Step 05d: digital skills and online banking among adult internet users (ECV 2025).

The ECV 2025 questionnaire sends persons who do not use the internet past the skills question (question 3 of the
chapter on information and communication technologies: "Si en preguntas 1a, 1b, 1c y 2 responde opción 5, pase a
7"), so the skills score (P1710S1-S12, 0-12) is observed for internet users only. This step therefore estimates the
skills analysis among internet users, in the conditional part only, and leaves the main two-part model for all
adults unchanged.

1. Availability of the skills score among adults, by territory and internet use (unweighted counts).
2. Weighted logit models of online banking among internet users (part 2 of step 05), with the covariates of the
   main model and, in turn, digital skills (0-12, scaled to 0-1), labour situation, and both. For the users of each
   territory T: observed conditional prevalence (model) = mean[pC(T)]; counterfactual = mean[pC(rest)], the same
   users with the territory indicator set to the rest of Colombia; adjusted conditional ratio = observed /
   counterfactual. Uncertainty: the PSU bootstrap of step 04 (Rao-Wu, 999 replicates, seed 20261120).

Outputs: results/02_two_part/skills_users/skills_availability.csv, conditional_ratios.csv, meta.json

Usage (from the repository root):  python src/05d_skills_users.py [--reps 999]
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tp", ROOT / "src" / "05_two_part.py")
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)
dec = tp.dec
MODELS = {"main": (False, False), "skills": (True, False), "labour": (False, True), "labour_skills": (True, True)}
SCOPES = ("Cesar", "Rest of the Caribbean", "Caribbean")


def ratios(d, w):
    """Adjusted conditional ratios among internet users for every model; d and w are restricted to users."""
    out = {}
    for name, (skills, labour) in MODELS.items():
        tp.LABOUR = labour
        X = tp.design(d)
        if skills:
            X["skills"] = d["skills"].to_numpy(dtype=float) / 12.0
        m = tp.fit(X, d["B"].to_numpy(), w)
        for t in SCOPES:
            mask = (d.territory != "Rest of Colombia").to_numpy() if t == "Caribbean" else (d.territory == t).to_numpy()
            Xt = X[mask]
            p_t = m.predict_proba(Xt.to_numpy())[:, 1]
            p_r = m.predict_proba(tp.set_territory(Xt, "Rest of Colombia").to_numpy())[:, 1]
            obs, cf = tp.wmean(p_t, w[mask]), tp.wmean(p_r, w[mask])
            out[(name, t)] = {"P_BU_model": obs, "P_BU_cf_rest": cf, "adj_ratio_BU": obs / cf}
    tp.LABOUR = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=999)
    a = ap.parse_args()
    t0 = time.time()
    out = ROOT / "results" / "02_two_part" / "skills_users"
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(ROOT / "data" / "processed" / "analytic_15plus.parquet")
    d = d[d.age >= 18].reset_index(drop=True)
    av = (d.assign(user=np.where(d.U == 1, "Internet users", "Non-users"), observed=d.skills.notna())
          .groupby(["territory", "user"])["observed"].agg(adults="size", with_score="sum").reset_index())
    av["without_score"] = av["adults"] - av["with_score"]
    av.to_csv(out / "skills_availability.csv", index=False)
    print(av.to_string())
    u = d[d.U == 1].reset_index(drop=True)
    assert u.skills.notna().all()
    w = u["w"].to_numpy()
    w = w / w.mean()
    point = ratios(u, w)
    rng = np.random.default_rng(dec.SEED)
    full = d  # the replicate multipliers are drawn for the PSUs of the whole adult sample, as in step 05
    psu_stratum = full.groupby("psu")["stratum"].first()
    psu_index, mult = dec.replicate_multipliers(psu_stratum, rng)
    pos = pd.Series(np.arange(len(psu_index)), index=psu_index)[u["psu"]].to_numpy()
    reps = []
    for r in range(a.reps):
        wr = w * mult[r, pos]
        keep = wr > 0
        reps.append(ratios(u[keep].reset_index(drop=True), wr[keep]))
        if (r + 1) % 100 == 0:
            print(f"replicate {r + 1}/{a.reps} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    rows = []
    for (name, t), est in point.items():
        for k, v in est.items():
            vals = np.array([rp[(name, t)][k] for rp in reps], dtype=float)
            rows.append({"model": name, "territory": t, "quantity": k, "estimate": v,
                         "lo": np.nanquantile(vals, 0.025), "hi": np.nanquantile(vals, 0.975)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "conditional_ratios.csv", index=False)
    meta = {"n_adults": int(len(d)), "n_users": int(len(u)), "reps": a.reps, "seed": dec.SEED,
            "runtime_min": round((time.time() - t0) / 60, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    print(res[res.quantity == "adj_ratio_BU"].round(3).to_string())


if __name__ == "__main__":
    main()
