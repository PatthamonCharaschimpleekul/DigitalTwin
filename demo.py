"""Streamlit page for the study: organ-level scores from the trained graph-and-patient model, and the paper's results.

Run `python train.py --splits temporal --models graph_patient --seeds 1 --out metrics_demo.csv --save-preds` once to
create the model file, then `streamlit run demo.py`."""
import numpy as np
import pandas as pd
import streamlit as st
import torch

from build_dataset import ORGANS
from train import DATA, DEVICE, RESULTS, Net, load, predict, split_train_val, splits, tensors

MODEL = RESULTS / "preds" / "temporal_0_graph_patient.pt"
ORGAN_NAMES = {"GI": "Gastrointestinal", "Nervous": "Nervous system"}
NAMES = {"global_rate": "Global rate", "drug_frequency": "Drug frequency", "patient_only": "Patient only",
         "logistic_regression": "Logistic regression", "gradient_boosting": "Gradient boosting",
         "structure_knn": "Structure k-NN", "drug_id": "Drug ID", "drug_id_patient": "Drug ID + patient",
         "graph": "Graph", "graph_patient": "Graph + patient", "fp_patient": "Fingerprint + patient",
         "graph_fp_patient": "Graph + fingerprint + patient", "atc_only_patient": "ATC class only + patient",
         "graph_random_patient": "Shuffled targets + patient"}

st.set_page_config(page_title="Organ-level ADE prediction", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading reports, graph and model…")
def setup():
    p, g, n_drugs, has_target = load()
    g = g.to(DEVICE)
    (_, tr, _), = splits(p, 0, has_target, ["temporal"])
    fit_rows, _ = split_train_val(tr, 0)  # train.run normalizes patient inputs on this subset
    net = Net("graph_patient", g, n_drugs).to(DEVICE)
    net.load_state_dict(torch.load(MODEL, map_location=DEVICE))
    net.eval()
    info = pd.read_csv(DATA / "drugs.csv").drop_duplicates("parent_chembl_id").set_index("parent_chembl_id")
    counts = p.groupby(["drug_id", "d"]).size().reset_index(name="reports")
    counts["name"] = counts.drug_id.map(info.generic_name).fillna(counts.drug_id).str.title()
    counts["atc"] = counts.drug_id.map(info.atc).fillna("")
    genes = pd.read_csv(DATA / "drug_gene.csv").groupby("drug_id").gene_symbol.apply(lambda s: sorted(set(s)))
    rate = tr.groupby("drug_id")[ORGANS].agg(["mean", "size"])
    return net, fit_rows, counts.sort_values("name").reset_index(drop=True), genes, tr, rate


def scores(net, fit_rows, d, age, sex, weight, n_other):
    row = pd.DataFrame({"age_years": age, "sex": sex, "weight_kg": weight, "n_other_drugs": n_other, "d": d}, index=[0])
    for o in ORGANS:
        row[o] = 0  # labels are unused at prediction time; tensors() expects the columns
    x, dd, _ = tensors(row, fit_rows)
    return predict(net, x, dd)


def results_tables():
    m = pd.concat([pd.read_csv(RESULTS / "metrics.csv"), pd.read_csv(RESULTS / "metrics_ablation.csv")])
    m = m[m.model.isin(NAMES)].drop_duplicates(["split", "model", "seed"])
    t = m.groupby(["model", "split"]).auroc.mean().unstack()[["temporal", "cold_drug", "atc_class"]]
    t = t.reindex([k for k in NAMES if k in t.index])  # paper's row order
    t.index = t.index.map(NAMES).rename("Model")
    t.columns = ["Temporal (seen drugs)", "Cold drug (unseen drugs)", "ATC class (unseen classes)"]
    b = pd.read_csv(RESULTS / "bootstrap.csv")
    b["comparison"] = b.a.map(NAMES) + "  vs  " + b.b.map(NAMES)
    b["split"] = b.split.map({"cold_drug": "Cold drug", "atc_class": "ATC class"})
    b["95% interval"] = b.apply(lambda r: f"{r.boot_ci95_low:+.3f} to {r.boot_ci95_high:+.3f}", axis=1)
    return t, b[["split", "comparison", "mean_diff", "95% interval"]].rename(columns={"mean_diff": "AUROC difference"})


st.title("Organ-level adverse drug event prediction")
st.caption("Patient-conditioned graph learning on 309,424 FAERS reports from 2025 · research prototype, not for clinical use")

if not MODEL.exists():
    st.error(f"Model file not found: {MODEL}. Run the training command in this file's docstring first.")
    st.stop()
net, fit_rows, drugs, genes, tr, rate = setup()

tab_pred, tab_res, tab_about = st.tabs(["Predict", "Results", "About and limitations"])

with tab_pred:
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Drug and patient")
        default = int(drugs.index[drugs.name == "Metformin"][0]) if (drugs.name == "Metformin").any() else 0
        i = st.selectbox("Primary suspect drug", drugs.index, index=default,
                         format_func=lambda k: f"{drugs.name[k]} ({drugs.reports[k]:,} reports)")
        drug = drugs.loc[i]
        age = st.slider("Age (years)", 0, 100, 60)
        sex = st.radio("Sex", ["F", "M", "Unknown"], horizontal=True)
        weight = st.number_input("Weight (kg, 0 = not reported)", 0.0, 250.0, 0.0, step=1.0)
        n_other = st.slider("Number of other reported drugs", 0, 30, 2)
        w = np.nan if weight == 0 else weight
        sx = None if sex == "Unknown" else sex
        st.markdown(f"**ATC:** {drug.atc.replace(';', ', ') or 'none'}")
        tg = genes.get(drug.drug_id, [])
        st.markdown(f"**Graph targets ({len(tg)}):** {', '.join(tg[:15]) + (' …' if len(tg) > 15 else '') if tg else 'none, so the graph sees only its ATC class'}")

    with right:
        prob = scores(net, fit_rows, drug.d, age, sx, w, n_other)[0]
        hist = rate.loc[drug.drug_id] if drug.drug_id in rate.index else None
        table = pd.DataFrame({
            "Organ system": [ORGAN_NAMES.get(o, o) for o in ORGANS],
            "Model score": prob,
            "This drug's rate in training reports": [hist[(o, "mean")] for o in ORGANS] if hist is not None else np.nan,
            "All training reports": tr[ORGANS].mean().values,
        }).set_index("Organ system")
        st.subheader("Share of reports mentioning each organ system")
        st.bar_chart(table[["Model score", "All training reports"]], horizontal=True, stack=False, height=340)
        n_hist = int(hist[(ORGANS[0], "size")]) if hist is not None else 0
        st.dataframe(table.style.format("{:.1%}"), width="stretch")
        st.caption(f"{n_hist:,} training reports (2025 Q1–Q3) for this drug. Scores estimate how often an organ "
                   "appears in a report about this drug and patient, among reports with at least one organ-mapped "
                   "event. They are not risks: FAERS has no exposure denominator.")

        st.subheader("How much do patient attributes move the scores?")
        ages = np.arange(5, 96, 5)
        sweep = np.vstack([scores(net, fit_rows, drug.d, a, sx, w, n_other)[0] for a in ages])
        st.line_chart(pd.DataFrame(sweep, index=ages, columns=[ORGAN_NAMES.get(o, o) for o in ORGANS]), height=260,
                      x_label="Age (years)", y_label="Model score")
        st.caption("Within-drug AUROC in the study was 0.56–0.59: patient attributes shift scores a little but do "
                   "not separate individual patients well.")

with tab_res:
    t, b = results_tables()
    st.subheader("Macro AUROC, mean of ten seeds")
    st.dataframe(t.style.format("{:.3f}", na_rep="–"), width="stretch")
    st.caption("–: needs report history, so not evaluated on held-out drugs.")
    st.markdown("- **Seen drugs:** report history is the strongest signal; the graph adds nothing.\n"
                "- **Unseen drugs:** the graph beats every comparator, and the gain disappears when drug–target edges are shuffled.\n"
                "- **Unseen therapeutic classes:** the graph still beats patient features but only matches gradient boosting and drug class alone.")
    st.subheader("Paired differences with drug-level bootstrap intervals")
    st.dataframe(b.style.format({"AUROC difference": "{:+.3f}"}), width="stretch", hide_index=True)

with tab_about:
    st.markdown("""
**Data.** FAERS 2025 quarterly reports matched to ChEMBL approved small molecules; a drug–gene–organ graph from
ChEMBL mechanisms (human targets) and curated CTD gene–disease associations. Organ labels come from a keyword
mapping of MedDRA preferred terms.

**Model on this page.** The graph-and-patient model trained on the temporal split (2025 Q1–Q3, seed 0), test
macro AUROC 0.720 on 2025 Q4. Drugs without graph targets are represented by their ATC class only.

**Limitations.** FAERS is voluntary, has no exposure denominator and cannot establish causality. Reports are
subject to under-reporting and stimulated reporting, and patient attributes may act as proxies for the indication.
Only the primary suspect drug is modeled, so drug–drug interactions are not covered. Labels were not reviewed by
clinicians. **This is a research prototype and must not guide treatment.**
""")
