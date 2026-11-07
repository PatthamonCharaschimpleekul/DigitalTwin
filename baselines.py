"""Classical comparators (logistic regression, gradient boosting, structure k-NN) on train.py's exact splits.

Rows append to results/baselines_partial.csv as each fit finishes; rerun resumes where it stopped.
Baseline rows already in results/metrics.csv are reused, so raising --seeds only runs the new seeds."""
import argparse

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_dataset import ORGANS
from train import DATA, RESULTS, load, scores, splits, tensors, within_drug_auroc

RDLogger.DisableLog("rdApp.*")
MODELS = ["logistic_regression", "gradient_boosting", "structure_knn"]
ML = ["logistic_regression", "gradient_boosting"]
FP_BITS, K = 1024, 10
COLS = ["split", "model", "seed", "auroc", "auprc"] + [f"auroc_{o}" for o in ORGANS] + ["within_drug_auroc"]


def fingerprints(p):
    drugs = pd.read_csv(DATA / "drugs.csv").drop_duplicates("parent_chembl_id").set_index("parent_chembl_id")
    cnt_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
    bit_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    cnt, bit = {}, {}
    for did in pd.unique(p.drug_id):
        smi = drugs.smiles.get(did)
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        cnt[did] = cnt_gen.GetCountFingerprintAsNumPy(mol).astype(np.float32) if mol else np.zeros(256, np.float32)
        bit[did] = bit_gen.GetFingerprint(mol) if mol else None
    frac = np.mean([v is not None for v in bit.values()])
    assert frac > 0.9, f"only {frac:.2%} of report drugs parsed as SMILES"
    return cnt, bit


def features(df, fit, atc, cnt):
    x = tensors(df, fit)[0].cpu().numpy()
    codes, uniq = pd.factorize(df.drug_id)  # stack once per unique drug (~1k), not per report (~300k)
    return np.column_stack([x, atc[df.d.values], np.stack([cnt[u] for u in uniq])[codes]]).astype(np.float32)


def knn_probs(tr, te, bit):
    rate = tr.groupby("drug_id")[ORGANS].mean()
    ids = [d for d in rate.index if bit.get(d) is not None]
    fps, R = [bit[d] for d in ids], rate.loc[ids].values
    glob = tr[ORGANS].mean().values
    pred = {}
    for did in pd.unique(te.drug_id):
        q = bit.get(did)
        if q is None:
            pred[did] = glob
            continue
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(q, fps))
        top = np.argsort(sims)[::-1][:K]
        top = top[sims[top] > 0]
        pred[did] = (R[top] * sims[top, None]).sum(0) / sims[top].sum() if len(top) else glob
    out = np.clip(np.stack([pred[i] for i in te.drug_id]), 0.0, 1.0)  # clip: 1-ulp overshoot of the convex combination
    assert np.isfinite(out).all()
    return out


def ml_probs(name, Xtr, ytr, Xte, seed):
    cols = []
    for j in range(ytr.shape[1]):
        model = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)) if name == "logistic_regression"
                 else HistGradientBoostingClassifier(random_state=seed))
        cols.append(model.fit(Xtr, ytr[:, j]).predict_proba(Xte)[:, 1])
    return np.column_stack(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)  # match train.py's run (seeds 0-9)
    args = ap.parse_args()
    p, g, n_drugs, has_target = load()
    atc = g["drug"].x.numpy()
    cnt, bit = fingerprints(p)
    part = RESULTS / "baselines_partial.csv"
    old = pd.read_csv(RESULTS / "metrics.csv")
    prev = old[old.model.isin(MODELS)].to_dict("records")  # seeds finished in an earlier run
    partial = pd.read_csv(part).to_dict("records") if part.exists() and part.stat().st_size else []
    keyed = {(r["split"], r["model"], int(r["seed"])): r for r in prev + partial}
    rows, done = list(keyed.values()), set(keyed)

    def save(row):
        rows.append(row)
        done.add((row["split"], row["model"], row["seed"]))
        # reindex to fixed columns: scores() omits degenerate organs, and a short row would shift columns on resume
        pd.DataFrame([row]).reindex(columns=COLS).to_csv(part, mode="a", header=not part.exists(), index=False)
        print(row, flush=True)

    for seed in range(args.seeds):
        for split, tr, te in splits(p, seed, has_target):
            y = te[ORGANS].values.astype(np.float32)
            if (split, "structure_knn", seed) not in done:
                save({"split": split, "model": "structure_knn", "seed": seed,
                      **scores(y, knn_probs(tr, te, bit)), "within_drug_auroc": 0.5})
            if not all((split, m, seed) in done for m in ML):
                Xtr, Xte = features(tr, tr, atc, cnt), features(te, tr, atc, cnt)
                ytr = tr[ORGANS].values.astype(np.float32)
                for name in ML:
                    if (split, name, seed) in done:
                        continue
                    prob = ml_probs(name, Xtr, ytr, Xte, seed)
                    save({"split": split, "model": name, "seed": seed, **scores(y, prob),
                          "within_drug_auroc": within_drug_auroc(y, prob, te.d.values)})
                del Xtr, Xte

    new = pd.DataFrame(rows)
    new = new[new.seed < args.seeds]
    assert len(new) == args.seeds * 2 * len(MODELS) and set(new.columns) == set(old.columns), len(new)
    out = pd.concat([old[~old.model.isin(MODELS)], new], ignore_index=True)
    out.to_csv(RESULTS / "metrics.csv", index=False)
    summary = out.groupby(["split", "model"])[["auroc", "auprc", "within_drug_auroc"]].agg(["mean", "std"]).round(4)
    summary.to_csv(RESULTS / "summary.csv")
    out.groupby(["split", "model"])[[f"auroc_{o}" for o in ORGANS]].mean().round(4).to_csv(RESULTS / "per_organ.csv")
    part.unlink()
    print(summary.to_string())


if __name__ == "__main__":
    main()
