# Patient-Conditioned Graph Learning for Organ-Level Adverse Drug Event Prediction

Code, results and paper source for a study that predicts which of eight organ systems (liver, heart, nervous,
kidney, lung, gastrointestinal, skin, blood) appear in an FDA Adverse Event Reporting System (FAERS) report, given
the drug and basic patient attributes. A drug–gene–organ graph built from ChEMBL mechanisms and curated CTD
gene–disease associations is compared against drug identity, chemical structure and patient attributes.

Paper: [`paper/main.tex`](paper/main.tex) ([PDF](paper/main.pdf)).

## Main results

Macro AUROC, mean over ten seeds (309,424 FAERS reports from 2025, 1,167 drugs).

| Model | Temporal (seen drugs) | Cold drug (unseen drugs) | ATC class (unseen classes) |
|---|---|---|---|
| Patient features only | 0.592 | 0.583 | 0.577 |
| Gradient boosting (patient + ATC + fingerprint) | 0.731 | 0.625 | **0.595** |
| Drug identity + patient | **0.732** | – | – |
| Graph + patient | 0.720 | **0.643** | 0.593 |
| Graph + fingerprint + patient | 0.730 | 0.637 | 0.590 |

- **Seen drugs:** report history is the strongest signal. The graph adds nothing.
- **Unseen drugs:** the graph beats every comparator, including gradient boosting, by 0.018 AUROC (95% drug-level
  bootstrap interval 0.011–0.024). The gain disappears when drug–target edges are shuffled. It survives removing CTD
  associations that could encode drug knowledge.
- **Unseen therapeutic classes:** the graph still beats patient features (+0.017, 0.004–0.029) but only matches
  gradient boosting and an ATC-class-only model.
- **Within-drug AUROC** (ranking patients on the same drug) stays at 0.56–0.59. This is a research prototype and
  must not guide treatment.

Full tables are in [`results/`](results): per-seed metrics (`metrics*.csv`), seed-paired differences
(`paired.csv`) and drug-level bootstrap intervals (`bootstrap.csv`).

## Reproducing the results

Requires Python 3.12 and an NVIDIA GPU (training asserts CUDA).

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # .venv/bin/pip on Linux/macOS
```

**1. Inputs.** Put these in `data/raw/`:
- FAERS quarterly ASCII files `faers_ascii_2025q1.zip` … `faers_ascii_2025q4.zip` from the
  [FDA FAERS download page](https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html)
- `CTD_curated_genes_diseases.csv.gz` from [CTD downloads](https://ctdbase.org/downloads/)

**2. Build the dataset.**

```bash
python fetch_chembl.py      # approved drugs, mechanisms and human targets from the ChEMBL API -> data/processed/
python build_dataset.py     # patient table, drug-gene and gene-organ edges (full and reduced CTD graph)
```

**3. Train and evaluate.** Each command appends to the result files, so they can be rerun one at a time.

```bash
python train.py                                                       # main neural models, all three splits
python train.py --models graph_random_patient,graph_no_atc_patient,atc_only_patient --out metrics_ablation.csv
python train.py --models fp_patient,graph_fp_patient --save-preds     # fingerprint variants
python train.py --splits cold_drug,atc_class --models patient_only,graph_patient,atc_only_patient --out metrics_preds.csv --save-preds
python train.py --graph gene_organ_strict.csv --splits cold_drug,atc_class --models graph,graph_patient --out metrics_strict.csv
python baselines.py                                                   # logistic regression, gradient boosting, structure k-NN
python bootstrap.py                                                   # drug-level bootstrap intervals -> results/bootstrap.csv
python paired.py                                                      # seed-paired differences -> results/paired.csv
```

The whole sequence takes roughly an hour on one laptop GPU, most of it in `baselines.py`. Training on the GPU is not bit-for-bit
deterministic: single seeds can move by up to about 0.009 AUROC on the ATC-class split, while ten-seed means move
by less than 0.001.

## Repository layout

| Path | Purpose |
|---|---|
| `fetch_chembl.py`, `build_dataset.py` | Data download and preprocessing |
| `train.py` | Splits, graph model, neural comparators and ablations |
| `baselines.py` | Classical comparators on the same splits |
| `bootstrap.py`, `paired.py` | Interval estimates reported in the paper |
| `results/` | Per-seed metrics, summaries and run logs |
| `paper/` | LaTeX source and PDF |
| `demo.py` | Web page for the trained model and results, see below |
| `app.py` and its CSV files | Earlier visual demo, see below |

## Study web page (`demo.py`)

A Streamlit page that runs the trained graph-and-patient model (temporal split, seed 0, test AUROC 0.720). Pick a
drug and patient attributes to see the eight organ scores next to the drug's rate in training reports, and how the
scores change with age. Other tabs show the paper's results and limitations.

```bash
python train.py --splits temporal --models graph_patient --seeds 1 --out metrics_demo.csv --save-preds   # creates the model file
streamlit run demo.py
```

## Earlier visual demo (`app.py`)

`app.py` is a Streamlit dashboard with a 3D body model, built before the study above. **It does not use the trained
models or any result in this repository.** Its organ alerts come from a fixed placeholder rule that picks an enzyme
from the digits in a drug's PubChem ID, and it flags a cross-interaction whenever two different drugs are selected.
Treat it as an interface mock-up only.

```bash
streamlit run app.py
```

## Limitations

FAERS is voluntary, has no exposure denominator and cannot establish causality. Organ labels come from a keyword
mapping of MedDRA preferred terms (in `build_dataset.py`), not licensed MedDRA groupings, and were not reviewed by
clinicians. Only the primary suspect drug is modeled. See the paper's Discussion section for the full list.
