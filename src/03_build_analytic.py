"""Step 03: analytic file of persons aged 15+ (adults 18+ is the main population and 15+ a sensitivity analysis).

Merges the ECV 2025 modules by DIRECTORIO + SECUENCIA_P + ORDEN (TIC, composition, education, design) and by
DIRECTORIO (dwelling: department and area). Variables:
- U = internet use (P1084 in 1-4); U_weekly = at least weekly (P1084 in 1-2) [sensitivity]
- B = online banking (P1083S5 = 1), defined only for internet users (0 for users who did not select it; non-users
  are B = 0 in the unconditional outcome)
- groups: territory (Cesar, rest of the Caribbean, rest of Colombia; and the Caribbean without San Andrés),
  area (CLASE), sex (P6020), age group, education (P8587), ethnicity (P6080)
- P1929 = main reason for not using the internet
- design: FEX_C, ESTRATO2020, PSU = MPIO:SEGMENTO

Output: data/processed/analytic_15plus.parquet, build_checks.json

Usage (from the repository root):  python src/03_build_analytic.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
KEY = ["DIRECTORIO", "SECUENCIA_P", "ORDEN"]
CARIBE = {"08", "13", "20", "23", "44", "47", "70", "88"}


def read(name, cols):
    return pd.read_csv(RAW / f"ecv2025_{name}.csv", sep=";", usecols=lambda c: c in cols, dtype=str, encoding="latin-1")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    tic = read("tic", KEY + ["FEX_C", "P1084", "P1083S5", "P1929"])
    com = read("composicion_hogar", KEY + ["P6020", "P6040", "P6080"])
    edu = read("educacion", KEY + ["P8587", "P8586", "P1088"])
    viv = read("vivienda", ["DIRECTORIO", "P1_DEPARTAMENTO", "CLASE"]).drop_duplicates("DIRECTORIO")
    dis = read("diseno_muestral", KEY + ["ESTRATO2020", "SEGMENTO", "MPIO"])
    d = tic.merge(com, on=KEY, validate="1:1").merge(edu, on=KEY, validate="1:1") \
           .merge(viv, on="DIRECTORIO", validate="m:1").merge(dis, on=KEY, validate="1:1")
    d["w"] = pd.to_numeric(d["FEX_C"].str.replace(",", "."), errors="coerce")
    d["age"] = pd.to_numeric(d["P6040"], errors="coerce")
    d = d[d["age"] >= 15].copy()
    d["U"] = d["P1084"].isin(["1", "2", "3", "4"]).astype(int)
    d["U_weekly"] = d["P1084"].isin(["1", "2"]).astype(int)
    d["B"] = ((d["U"] == 1) & (d["P1083S5"] == "1")).astype(int)
    d["B_weekly"] = ((d["U_weekly"] == 1) & (d["P1083S5"] == "1")).astype(int)
    dept = d["P1_DEPARTAMENTO"].str.zfill(2)
    d["dept"] = dept
    d["territory"] = np.where(dept == "20", "Cesar", np.where(dept.isin(CARIBE), "Rest of the Caribbean",
                                                                "Rest of Colombia"))
    d["caribbean"] = np.where(dept.isin(CARIBE), "Caribbean", "Rest of Colombia")
    d["caribbean_no_sai"] = np.where(dept.isin(CARIBE - {"88"}), "Caribbean", np.where(dept == "88", "San Andrés",
                                                                                     "Rest of Colombia"))
    d["area"] = np.where(d["CLASE"] == "1", "Urban (cabecera)", "Rural (centros poblados y rural disperso)")
    d["sex"] = d["P6020"].map({"1": "Men", "2": "Women"})
    d["age_group"] = pd.cut(d["age"], [14, 17, 29, 44, 59, 200], labels=["15-17", "18-29", "30-44", "45-59", "60+"]).astype(str)
    # highest level attained (P8587) for those not studying; current students (P8586 = 1) are not asked P8587 and
    # are classified by the level in which they are enrolled (P1088): 1-2 primary, 3-4 secondary, 5-8 tertiary
    e = pd.to_numeric(d["P8587"], errors="coerce")
    cur = pd.to_numeric(d["P1088"], errors="coerce")
    d["education"] = np.select([e <= 3, e.between(4, 5), e >= 6, cur <= 2, cur.between(3, 4), cur >= 5],
                               ["Primary or less", "Secondary", "Technical or higher", "Primary or less", "Secondary",
                                "Technical or higher"], default="Unknown")
    d["student"] = (d["P8586"] == "1").astype(int)
    d["ethnicity"] = d["P6080"].map({"1": "Indigenous", "5": "Black or Afro-Colombian", "6": "None"}).fillna("Other")
    d["reason_nonuse"] = d["P1929"].map({"1": "Too expensive", "2": "Not necessary", "3": "Does not know how to use it",
                                         "4": "No coverage", "5": "Not allowed", "6": "Security or privacy",
                                         "7": "Other"})
    d["stratum"] = d["ESTRATO2020"]
    d["psu"] = d["MPIO"] + ":" + d["SEGMENTO"]
    keep = ["DIRECTORIO", "SECUENCIA_P", "ORDEN", "w", "age", "U", "U_weekly", "B", "B_weekly", "dept", "territory",
            "caribbean", "caribbean_no_sai", "area", "sex", "age_group", "education", "student", "ethnicity", "reason_nonuse",
            "stratum", "psu"]
    d = d[keep]
    d.to_parquet(OUT / "analytic_15plus.parquet", index=False)
    a = d[d.age >= 18]
    chk = {"n_15plus": int(len(d)), "n_18plus": int(len(a)), "missing_weight": int(d.w.isna().sum()),
           "education_unknown_18plus": int((a.education == "Unknown").sum()),
           "ethnicity_other_18plus": int((a.ethnicity == "Other").sum()),
           "sex_missing": int(d.sex.isna().sum()), "users_18plus": int(a.U.sum()), "ebank_18plus": int(a.B.sum()),
           "B_without_U": int(((d.B == 1) & (d.U == 0)).sum()),
           "nonusers_with_reason_18plus": int(((a.U == 0) & a.reason_nonuse.notna()).sum()),
           "nonusers_18plus": int((a.U == 0).sum()), "n_strata": int(d.stratum.nunique()), "n_psu": int(d.psu.nunique()),
           "territory_n_18plus": a.territory.value_counts().to_dict()}
    assert chk["B_without_U"] == 0 and chk["missing_weight"] == 0
    (OUT / "build_checks.json").write_text(json.dumps(chk, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(chk, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
