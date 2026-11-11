"""Train and compare organ-level adverse event models on FAERS with a drug-gene-organ graph."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from build_dataset import ORGANS

ROOT = Path(__file__).parent
DATA, RESULTS = ROOT / "data" / "processed", ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # main() still requires a GPU; demo.py runs on CPU
ATC_LEVELS = list("ABCDGHJLMNPRSV")
WARM_MODELS = ["patient_only", "drug_id", "drug_id_patient", "graph", "graph_patient"]
COLD_MODELS = ["patient_only", "graph", "graph_patient"]  # an ID embedding is untrained for unseen drugs
ABLATION_MODELS = ["graph_random_patient", "graph_no_atc_patient", "atc_only_patient"]
FP_BITS = 256  # same Morgan count fingerprint (radius 2) the classical baselines use
PRED_DIR = None  # set by --save-preds; run() then stores test predictions for bootstrap.py
RDLogger.DisableLog("rdApp.*")


def ablate(kind, g, seed):
    """graph_random: drug-gene endpoints shuffled (each drug keeps its degree; gene marginals drift, upgrade to
    degree-preserving swaps if reviewers ask). graph_no_atc: drug ATC input replaced by ones. atc_only: target edges gone."""
    gg = copy.deepcopy(g)
    if "random" in kind:
        e1 = gg["drug", "targets", "gene"].edge_index.cpu().numpy()
        e1 = np.vstack([e1[0], np.random.default_rng(seed).permutation(e1[1])])
        t = torch.tensor(e1, device=DEVICE)
        gg["drug", "targets", "gene"].edge_index = t
        gg["gene", "rev_targets", "drug"].edge_index = t.flip(0)
    if "no_atc" in kind:
        gg["drug"].x = torch.ones_like(gg["drug"].x)
    if kind.startswith("atc_only"):
        z = torch.zeros((2, 0), dtype=torch.long, device=DEVICE)
        gg["drug", "targets", "gene"].edge_index = z
        gg["gene", "rev_targets", "drug"].edge_index = z
    return gg


def load(gene_organ="gene_organ.csv"):
    p = pd.read_csv(DATA / "patients.csv.gz")
    drugs = pd.read_csv(DATA / "drugs.csv").drop_duplicates("parent_chembl_id").set_index("parent_chembl_id")
    dg, go = pd.read_csv(DATA / "drug_gene.csv"), pd.read_csv(DATA / gene_organ)
    ids = sorted(set(p.drug_id) | set(dg.drug_id))
    genes = sorted(set(dg.gene_symbol) | set(go.gene_symbol))
    d_ix = {d: i for i, d in enumerate(ids)}
    g_ix = {g: i for i, g in enumerate(genes)}
    o_ix = {o: i for i, o in enumerate(ORGANS)}

    atc = torch.zeros(len(ids), len(ATC_LEVELS))
    for d, codes in drugs.atc.reindex(ids).fillna("").items():
        for c in filter(None, codes.split(";")):
            if c[0] in ATC_LEVELS:
                atc[d_ix[d], ATC_LEVELS.index(c[0])] = 1

    fp = np.zeros((len(ids), FP_BITS), np.float32)  # zeros for drugs without a parsable SMILES
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    for i, smi in enumerate(drugs.smiles.reindex(ids)):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol:
            fp[i] = gen.GetCountFingerprintAsNumPy(mol)

    g = HeteroData()
    g["drug"].x = atc
    g["drug"].fp = torch.tensor(np.log1p(fp))  # input of the *fp* models only; the graph convs never read it
    g["gene"].num_nodes = len(genes)
    g["organ"].num_nodes = len(ORGANS)
    e1 = torch.tensor(np.array([dg.drug_id.map(d_ix).values, dg.gene_symbol.map(g_ix).values]))
    e2 = torch.tensor(np.array([go.gene_symbol.map(g_ix).values, go.organ.map(o_ix).values]))
    g["drug", "targets", "gene"].edge_index = e1
    g["gene", "rev_targets", "drug"].edge_index = e1.flip(0)
    g["gene", "linked", "organ"].edge_index = e2
    g["organ", "rev_linked", "gene"].edge_index = e2.flip(0)

    p["d"] = p.drug_id.map(d_ix)
    atc3 = {d: ";".join(sorted({c[:4] for c in filter(None, codes.split(";"))}))
            for d, codes in drugs.atc.reindex(ids).fillna("").items()}
    p["atc3"] = p.drug_id.map(atc3)  # ATC level-3 classes of the report's drug, "" if none
    has_target = np.zeros(len(ids), bool)
    has_target[e1[0].unique().numpy()] = True
    return p, g, len(ids), has_target


def tensors(p, fit):
    age, wt = p.age_years, p.weight_kg
    x = np.column_stack([
        ((age - fit.age_years.mean()) / fit.age_years.std()).fillna(0), age.isna(),
        ((wt - fit.weight_kg.mean()) / fit.weight_kg.std()).fillna(0), wt.isna(),
        p.sex == "F", p.sex == "M", np.log1p(p.n_other_drugs),
    ]).astype(np.float32)
    return (torch.tensor(x, device=DEVICE), torch.tensor(p.d.values, device=DEVICE),
            torch.tensor(p[ORGANS].values, dtype=torch.float32, device=DEVICE))


class Net(nn.Module):
    def __init__(self, kind, g, n_drugs, dim=64):
        super().__init__()
        self.use_patient = "patient" in kind  # covers patient_only as well as *_patient
        self.use_graph = kind.startswith("graph") or kind.startswith("atc_only")
        self.use_id = kind.startswith("drug_id")
        if self.use_graph:
            self.g = g
            self.drug_in = nn.Linear(len(ATC_LEVELS), dim)
            self.gene = nn.Embedding(g["gene"].num_nodes, dim)
            self.organ = nn.Embedding(len(ORGANS), dim)
            self.convs = nn.ModuleList(
                HeteroConv({et: SAGEConv((dim, dim), dim) for et in g.edge_types}, aggr="sum") for _ in range(2))
        if self.use_id:
            self.drug = nn.Embedding(n_drugs, dim)
        self.patient = nn.Sequential(nn.Linear(7, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.use_fp = "fp" in kind  # fp_patient, graph_fp_patient: chemical structure as an extra drug input
        if self.use_fp:
            self.register_buffer("fp", g["drug"].fp)
            self.fp_in = nn.Sequential(nn.Linear(FP_BITS, dim), nn.ReLU())
        width = dim * (int(self.use_patient) + int(self.use_graph or self.use_id) + int(self.use_fp))
        self.head = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, len(ORGANS)))

    def drug_table(self):
        if self.use_id:
            return self.drug.weight
        if not self.use_graph:
            return None
        h = {"drug": self.drug_in(self.g["drug"].x), "gene": self.gene.weight, "organ": self.organ.weight}
        for conv in self.convs:
            h = {k: F.relu(v) for k, v in conv(h, self.g.edge_index_dict).items()}
        return h["drug"]

    def forward(self, x, d):
        table, parts = self.drug_table(), []
        if table is not None:
            parts.append(table[d])
        if self.use_fp:
            parts.append(self.fp_in(self.fp[d]))
        if self.use_patient:
            parts.append(self.patient(x))
        return self.head(torch.cat(parts, 1))


def predict(net, x, d):
    net.eval()
    with torch.no_grad():
        return torch.cat([torch.sigmoid(net(x[i:i + 65536], d[i:i + 65536]))
                          for i in range(0, len(x), 65536)]).cpu().numpy()


def fit(kind, g, n_drugs, train, val, seed, epochs, batch=8192):
    torch.manual_seed(seed)
    net = Net(kind, g, n_drugs).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    x, d, y = train
    best, best_state, stale = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(x), device=DEVICE).split(batch):
            loss = F.binary_cross_entropy_with_logits(net(x[idx], d[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv = torch.tensor(predict(net, val[0], val[1]), device=DEVICE).clamp(1e-6, 1 - 1e-6)
        v = F.binary_cross_entropy(pv, val[2]).item()
        if v < best:
            best, stale = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            stale += 1
            if stale >= 5:
                break
    net.load_state_dict(best_state)
    return net


def scores(y, p):
    keep = [j for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    per_organ = {f"auroc_{ORGANS[j]}": float(roc_auc_score(y[:, j], p[:, j])) for j in keep}
    return {"auroc": float(np.mean(list(per_organ.values()))),
            "auprc": float(np.mean([average_precision_score(y[:, j], p[:, j]) for j in keep])), **per_organ}


def within_drug_auroc(y, p, d, min_reports=200):
    """Discrimination among patients on the same drug; drug-level base rates cannot score here."""
    vals = []
    for drug in np.unique(d):
        m = d == drug
        if m.sum() < min_reports:
            continue
        vals += [roc_auc_score(y[m, j], p[m, j]) for j in range(y.shape[1]) if 20 <= y[m, j].sum() <= m.sum() - 20]
    return float(np.mean(vals)) if vals else float("nan")


def split_train_val(df, seed):
    m = np.random.default_rng(seed).random(len(df)) < 0.1
    return df[~m], df[m]


SPLITS = ("temporal", "cold_drug", "atc_class")


def splits(p, seed, has_target, which=SPLITS):
    """The exact held-out sets every model is compared on; baselines.py imports this too."""
    if "temporal" in which:
        yield "temporal", p[p.quarter != "2025q4"], p[p.quarter == "2025q4"]
    if "cold_drug" in which:
        pool = np.array(sorted(set(p.d[has_target[p.d.values]])))
        held = np.random.default_rng(1000 + seed).choice(pool, len(pool) // 5, replace=False)
        cold = p.d.isin(held)
        yield "cold_drug", p[~cold], p[cold]
    if "atc_class" in which:
        # hold out whole ATC level-3 classes, so no drug sharing a class with a test drug is trained on
        dc = p.drop_duplicates("d").set_index("d").atc3
        classes = sorted({c for s in dc[has_target[dc.index] & (dc != "")] for c in s.split(";")})
        held_c = set(np.random.default_rng(2000 + seed).choice(classes, len(classes) // 5, replace=False))
        touch = [d for d, s in dc.items() if s and held_c & set(s.split(";"))]
        inside = [d for d, s in dc.items() if s and set(s.split(";")) <= held_c]
        # test: drugs whose classes are all held out; train: drugs with no held-out class; the rest are dropped
        yield "atc_class", p[~p.d.isin(touch)], p[p.d.isin(inside) & has_target[p.d.values]]


def run(split, tr, te, kinds, g, n_drugs, seed, epochs):
    tr, va = split_train_val(tr, seed)
    train, val, test = tensors(tr, tr), tensors(va, tr), tensors(te, tr)
    y, d = test[2].cpu().numpy(), test[1].cpu().numpy()
    rows = []

    base = np.tile(tr[ORGANS].mean().values, (len(te), 1)) + np.random.default_rng(seed).normal(0, 1e-9, (len(te), len(ORGANS)))
    rows.append({"split": split, "model": "global_rate", "seed": seed, **scores(y, base), "within_drug_auroc": 0.5})
    if split == "temporal":
        rate = tr.groupby("d")[ORGANS].mean()
        freq = rate.reindex(te.d).fillna(tr[ORGANS].mean()).values
        rows.append({"split": split, "model": "drug_frequency", "seed": seed, **scores(y, freq), "within_drug_auroc": 0.5})

    if PRED_DIR:
        np.savez_compressed(PRED_DIR / f"{split}_{seed}_test.npz", y=y.astype(np.uint8), d=d)
    for kind in kinds:
        net = fit(kind, ablate(kind, g, seed), n_drugs, train, val, seed, epochs)
        p = predict(net, test[0], test[1])
        if PRED_DIR:
            np.savez_compressed(PRED_DIR / f"{split}_{seed}_{kind}.npz", p=p)
            torch.save(net.state_dict(), PRED_DIR / f"{split}_{seed}_{kind}.pt")  # demo.py loads temporal_0_graph_patient.pt
        rows.append({"split": split, "model": kind, "seed": seed, **scores(y, p),
                     "within_drug_auroc": within_drug_auroc(y, p, d)})
        print(rows[-1], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--models", default=None, help="comma list replacing WARM/COLD sets, for ablation runs")
    ap.add_argument("--out", default="metrics.csv")
    ap.add_argument("--splits", default=",".join(SPLITS), help="comma list of splits to run")
    ap.add_argument("--graph", default="gene_organ.csv", help="gene-organ edge file in data/processed")
    ap.add_argument("--save-preds", action="store_true", help="store test predictions in results/preds")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU required"
    RESULTS.mkdir(exist_ok=True)
    if args.save_preds:
        global PRED_DIR
        PRED_DIR = RESULTS / "preds"
        PRED_DIR.mkdir(exist_ok=True)

    p, g, n_drugs, has_target = load(args.graph)
    g = g.to(DEVICE)
    print(f"reports={len(p)} drugs={p.d.nunique()}", flush=True)

    warm = args.models.split(",") if args.models else WARM_MODELS
    cold_m = args.models.split(",") if args.models else COLD_MODELS
    out_f = RESULTS / args.out
    rows = []
    for seed in range(args.seeds):
        for split, tr, te in splits(p, seed, has_target, args.splits.split(",")):
            print(f"seed={seed} split={split} train={len(tr)} test={len(te)} test_drugs={te.d.nunique()}", flush=True)
            rows += run(split, tr, te, warm if split == "temporal" else cold_m, g, n_drugs, seed, args.epochs)
        df = pd.DataFrame(rows)
        if out_f.exists():  # merge, not clobber: this run's (split, model) rows replace prior ones, the rest survive
            old = pd.read_csv(out_f)
            ran = old.set_index(["split", "model"]).index.isin(df.set_index(["split", "model"]).index)
            df = pd.concat([old[~ran], df], ignore_index=True)
        df.to_csv(out_f, index=False)

    if args.out == "metrics.csv":  # merged runs regenerating summaries
        summary = df.groupby(["split", "model"])[["auroc", "auprc", "within_drug_auroc"]].agg(["mean", "std"]).round(4)
        summary.to_csv(RESULTS / "summary.csv")
        df.groupby(["split", "model"])[[f"auroc_{o}" for o in ORGANS]].mean().round(4).to_csv(RESULTS / "per_organ.csv")
        print(summary.to_string())
    (RESULTS / "run_config.json").write_text(json.dumps(vars(args)))


if __name__ == "__main__":
    main()
