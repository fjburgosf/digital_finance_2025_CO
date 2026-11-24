"""Step 01: prepare the ECV 2025 microdata (DANE) used by the analysis.

Download the CSV version of the seven modules listed below from the DANE microdata catalogue (ANDA, ECV 2025,
catalogue 905) and place the zip files, as delivered, in data/downloads/. This script extracts the CSV member of each
zip into data/raw/ with a fixed name and records the SHA-256 of every file in data/raw/checksums.json.
If you already have the CSV files, you can place them directly in data/raw/ with the names below.

    data/raw/ecv2025_tic.csv                 Tecnologías de información y comunicación
    data/raw/ecv2025_composicion_hogar.csv   Características y composición del hogar
    data/raw/ecv2025_vivienda.csv            Datos de la vivienda
    data/raw/ecv2025_educacion.csv           Educación
    data/raw/ecv2025_diseno_muestral.csv     Variables de diseño muestral
    data/raw/ecv2025_fuerza_trabajo.csv      Fuerza de trabajo
    data/raw/ecv2025_servicios_hogar.csv     Servicios del hogar

Usage (from the repository root):  python src/01_prepare_data.py
"""
import hashlib
import json
import shutil
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "data" / "downloads"
RAW = ROOT / "data" / "raw"
# target file -> keyword that identifies the module in the name of the zip file downloaded from ANDA
MODULES = {
    "ecv2025_tic.csv": "tecnologias",
    "ecv2025_composicion_hogar.csv": "composicion",
    "ecv2025_vivienda.csv": "vivienda",
    "ecv2025_educacion.csv": "educacion",
    "ecv2025_diseno_muestral.csv": "diseno",
    "ecv2025_fuerza_trabajo.csv": "fuerza",
    "ecv2025_servicios_hogar.csv": "servicios",
}


def plain(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    zips = sorted(DOWNLOADS.glob("*.zip")) if DOWNLOADS.exists() else []
    record = {}
    for name, key in MODULES.items():
        dst = RAW / name
        if not dst.exists():
            cand = [z for z in zips if key in plain(z.name) and (key != "vivienda" or "datos" in plain(z.name))]
            if not cand:
                raise SystemExit(f"Missing {name}: put the ANDA zip of the module in {DOWNLOADS} or the CSV in {RAW}.")
            with zipfile.ZipFile(cand[0]) as zf:
                member = next(m for m in zf.namelist() if m.lower().endswith(".csv"))
                with zf.open(member) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
            record[name] = {"source_zip": cand[0].name, "member": member}
        record.setdefault(name, {}).update({"bytes": dst.stat().st_size, "sha256": sha256(dst)})
        print(f"{name:34s} {record[name]['bytes']:>12d} {record[name]['sha256'][:16]}")
    (RAW / "checksums.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
