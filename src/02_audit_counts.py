"""Step 02: audit of the ECV 2025 microdata (unweighted counts only).

- Merge keys DIRECTORIO + SECUENCIA_P + ORDEN between the TIC, composition and education modules (unique, unmatched).
- Department and area from the dwelling module (DIRECTORIO); sample design variables.
- Counts of persons aged 18+ by department of the Caribbean region and area, of internet users (P1084 in 1-4) and of
  users who report online banking (P1083S5 = 1), to check the cell sizes available for the decomposition.
- Consistency: no non-user reports an internet activity; skip pattern of P1083S5 among users.

Output: results/00_audit/audit_counts.json

Usage (from the repository root):  python src/02_audit_counts.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "00_audit"
KEY = ["DIRECTORIO", "SECUENCIA_P", "ORDEN"]
CARIBE = {"08": "Atlántico", "13": "Bolívar", "20": "Cesar", "23": "Córdoba", "44": "La Guajira", "47": "Magdalena",
          "70": "Sucre", "88": "San Andrés"}


def read(name, cols):
    return pd.read_csv(RAW / f"ecv2025_{name}.csv", sep=";", usecols=lambda c: c in cols, dtype=str, encoding="latin-1")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    tic = read("tic", KEY + ["FEX_C", "P1084", "P1083S5", "P1083S1", "P1083S3", "P1929"])
    com = read("composicion_hogar", KEY + ["P6020", "P6040", "P6080"])
    edu = read("educacion", KEY + ["P8587"])
    viv = read("vivienda", ["DIRECTORIO", "P1_DEPARTAMENTO", "CLASE", "REGION"])
    dis = read("diseno_muestral", KEY + ["ESTRATO2020", "SEGMENTO", "MPIO"])
    rep = {"rows": {"tic": len(tic), "composicion": len(com), "educacion": len(edu), "vivienda": len(viv),
                    "diseno": len(dis)}}
    rep["unique_keys"] = {k: bool(not d.duplicated(KEY).any()) for k, d in (("tic", tic), ("composicion", com),
                                                                             ("educacion", edu), ("diseno", dis))}
    m = tic.merge(com, on=KEY, how="left", indicator="m_com").merge(edu, on=KEY, how="left", indicator="m_edu")
    m = m.merge(viv.drop_duplicates("DIRECTORIO"), on="DIRECTORIO", how="left", indicator="m_viv")
    m = m.merge(dis, on=KEY, how="left", indicator="m_dis")
    rep["unmatched_from_tic"] = {c: int((m[c] != "both").sum()) for c in ("m_com", "m_edu", "m_viv", "m_dis")}
    m["age"] = pd.to_numeric(m["P6040"], errors="coerce")
    m["user"] = m["P1084"].isin(["1", "2", "3", "4"])
    m["ebank"] = m["P1083S5"] == "1"
    rep["p1084_values"] = m["P1084"].value_counts(dropna=False).to_dict()
    rep["p1083s5_values_among_users"] = m.loc[m.user, "P1083S5"].value_counts(dropna=False).to_dict()
    rep["nonusers_with_any_activity"] = int((~m.user & (m[["P1083S1", "P1083S3", "P1083S5"]] == "1").any(axis=1)).sum())
    a = m[m.age >= 18].copy()
    a["dept"] = a["P1_DEPARTAMENTO"].str.zfill(2)
    rep["adults"] = {"n": int(len(a)), "users": int(a.user.sum()), "ebank": int(a.ebank.sum())}
    car = a[a.dept.isin(CARIBE)]
    tab = car.groupby(["dept", "CLASE"]).agg(n=("user", "size"), users=("user", "sum"), ebank=("ebank", "sum"))
    rep["caribe_by_dept_area"] = {f"{CARIBE[d]}|{'cabecera' if c == '1' else 'resto'}": {k: int(v) for k, v in r.items()}
                                  for (d, c), r in tab.iterrows()}
    rep["caribe_total"] = {"n": int(len(car)), "users": int(car.user.sum()), "ebank": int(car.ebank.sum())}
    rep["ethnicity_adults_caribe"] = car["P6080"].value_counts(dropna=False).to_dict()
    rep["n_strata"] = int(m["ESTRATO2020"].nunique())
    rep["n_psu"] = int((m["MPIO"] + ":" + m["SEGMENTO"]).nunique())
    (OUT / "audit_counts.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=int), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=int))


if __name__ == "__main__":
    main()
