# Internet use and online banking among adults in Colombia (ECV 2025): analysis code

This repository contains the analysis code used to study internet use and online banking among adults in Colombia
with the 2025 National Quality of Life Survey (Encuesta Nacional de Calidad de Vida, ECV) of the National
Administrative Department of Statistics (DANE). The analysis splits the online-banking gap between groups into two
stages, internet use and online banking among internet users, using the identity

    P(B) = P(U) × P(B | U)    so that    ln[P(B)g / P(B)r] = ln[P(U)g / P(U)r] + ln[P(B|U)g / P(B|U)r]

where U is internet use, B is online banking, g is a group and r its reference group. It also fits a two-part model
adjusted for covariates, with territorial counterfactuals for Cesar and the Caribbean region.

The repository includes only the analysis scripts. It does not include the survey microdata.

## Data

The microdata are not redistributed here. They are public and can be downloaded, after registration, from the DANE
microdata catalogue (ANDA): ECV 2025, catalogue 905, <https://microdatos.dane.gov.co/index.php/catalog/905>.

Download the CSV version of these five modules:

| Module (ANDA) | File created in `data/raw/` |
|---|---|
| Tecnologías de información y comunicación | `ecv2025_tic.csv` |
| Características y composición del hogar | `ecv2025_composicion_hogar.csv` |
| Datos de la vivienda | `ecv2025_vivienda.csv` |
| Educación | `ecv2025_educacion.csv` |
| Variables de diseño muestral | `ecv2025_diseno_muestral.csv` |

Put the zip files, as delivered by ANDA, in `data/downloads/` (step 01 extracts them), or put the CSV files directly
in `data/raw/` with the names of the table.

## Installation

Python 3.12 was used.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Running the analysis

Run the scripts from the repository root, in this order:

```bash
python src/01_prepare_data.py
python src/02_audit_counts.py
python src/03_build_analytic.py
python src/04_decomposition.py --tag main
python src/05_two_part.py --tag main
```

| Step | Script | What it does | Output |
|---|---|---|---|
| 01 | `01_prepare_data.py` | Extracts the five ECV modules and records their SHA-256 | `data/raw/*.csv`, `data/raw/checksums.json` |
| 02 | `02_audit_counts.py` | Checks merge keys and skip patterns; unweighted counts by department and area | `results/00_audit/audit_counts.json` |
| 03 | `03_build_analytic.py` | Builds the analytic file of persons aged 15 or more (outcomes, groups and design variables) | `data/processed/analytic_15plus.parquet`, `build_checks.json` |
| 04 | `04_decomposition.py` | Weighted prevalences of P(U), P(B\|U) and P(B), and exact log decomposition of the gaps, for Colombia, the Caribbean region and Cesar | `results/01_decomposition/<tag>/prevalences.csv`, `decomposition.csv`, `meta.json` |
| 05 | `05_two_part.py` | Weighted two-part logit model (U; B given U), territorial counterfactuals, average marginal effects and a Fairlie-type decomposition | `results/02_two_part/<tag>/counterfactuals.csv`, `coefficients.csv`, `fairlie.csv`, `meta.json` |

### Sensitivity analyses

```bash
python src/04_decomposition.py --tag pop15 --population 15
python src/04_decomposition.py --tag weekly --weekly
python src/04_decomposition.py --tag no_sai --caribbean-no-sai
```

They use, respectively, the population aged 15 or more, internet use at least once a week, and the Caribbean region
without San Andrés.

## Definitions

- Internet use (U): `P1084` in 1-4. At least weekly use: `P1084` in 1-2.
- Online banking (B): `P1083S5` = 1 among internet users; 0 for non-users in the unconditional outcome.
- Territory: Cesar (department 20), rest of the Caribbean (08, 13, 23, 44, 47, 70, 88) and rest of Colombia.
- Groups: area (`CLASE`), sex (`P6020`), age group (`P6040`), education (`P8587`; current students by the level
  in which they are enrolled, `P1088`) and ethnic self-identification (`P6080`).
- Survey design: weight `FEX_C`, strata `ESTRATO2020`, primary sampling units `MPIO:SEGMENTO`.

## Uncertainty and disclosure control

Confidence intervals come from a bootstrap of primary sampling units within strata (Rao-Wu rescaling, 999
replicates, seed 20261120), with percentile 95% intervals. In step 05 every model is refitted in each replicate; with
999 replicates this took about 13 minutes on a desktop computer (use `--reps` to change the number). Cells with fewer
than 30 unweighted persons are flagged as suppressed, and estimates with a coefficient of variation above 15% are
flagged, following the DANE recommendation.
