"""Drug-level cluster bootstrap intervals for paired macro-AUROC differences on the held-out-drug splits.

Each replicate resamples test drugs with replacement within every seed, recomputes both models' macro AUROC on the
same resample, and averages the difference over seeds. Neural predictions come from `train.py --save-preds`;
gradient boosting predictions are fitted here once and cached next to them."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from build_dataset import ORGANS
from train import RESULTS, load, splits

PRED = RESULTS / "preds"
SEEDS, B = range(10), 1000
PAIRS = [("graph_patient", "patient_only"), ("graph_patient", "gradient_boosting"),
         ("graph_patient", "atc_only_patient"), ("graph_fp_patient", "graph_patient"),
         ("graph_fp_patient", "gradient_boosting"), ("fp_patient", "gradient_boosting")]


def weighted_auc(groups, n_groups, y, w):
    """AUROC with row weights; groups ranks tied scores once so each replicate is two bincounts."""
    pos, neg = np.bincount(groups, w * y, n_groups), np.bincount(groups, w * (1 - y), n_groups)
    below = np.cumsum(neg) - neg
    return (pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum())


def gb_preds(split, seed, d, p, g, has_target, cache={}):
    f = PRED / f"{split}_{seed}_gradient_boosting.npz"
    if not f.exists():
        from baselines import features, fingerprints, ml_probs
        if "cnt" not in cache:
            cache["cnt"] = fingerprints(p)[0]
        (_, tr, te), = splits(p, seed, has_target, [split])
        assert (te.d.values == d).all(), "test rows differ from the saved neural predictions"
        atc = g["drug"].x.numpy()
        prob = ml_probs("gradient_boosting", features(tr, tr, atc, cache["cnt"]), tr[ORGANS].values.astype(np.float32),
                        features(te, tr, atc, cache["cnt"]), seed)
        np.savez_compressed(f, p=prob)
    return np.load(f)["p"]


def main():
    rows = []
    p = g = has_target = None
    for split in ["cold_drug", "atc_class"]:
        models = sorted({m for pair in PAIRS for m in pair})
        per_seed = []
        for seed in SEEDS:
            t = np.load(PRED / f"{split}_{seed}_test.npz")
            y, d = t["y"].astype(float), t["d"]
            preds = {}
            for m in models:
                if m == "gradient_boosting" and not (PRED / f"{split}_{seed}_{m}.npz").exists() and p is None:
                    p, g, _, has_target = load()
                preds[m] = gb_preds(split, seed, d, p, g, has_target) if m == "gradient_boosting" \
                    else np.load(PRED / f"{split}_{seed}_{m}.npz")["p"]
                assert preds[m].shape == y.shape, (split, seed, m)
            groups = {(m, j): np.unique(preds[m][:, j], return_inverse=True) for m in models for j in range(y.shape[1])}
            groups = {k: (inv.ravel(), len(u)) for k, (u, inv) in groups.items()}
            drugs, d_idx = np.unique(d, return_inverse=True)
            per_seed.append((y, d_idx, len(drugs), groups))

        rng = np.random.default_rng(0)
        draws = {pair: np.zeros(B + 1) for pair in PAIRS}  # index 0: unweighted point estimate
        for b in range(B + 1):
            diffs = {pair: [] for pair in PAIRS}
            for y, d_idx, n_drugs, groups in per_seed:
                w = np.ones(len(y)) if b == 0 else rng.multinomial(n_drugs, np.full(n_drugs, 1 / n_drugs))[d_idx].astype(float)
                organs = [j for j in range(y.shape[1]) if 0 < (w * y[:, j]).sum() < w.sum()]
                auc = {m: np.mean([weighted_auc(*groups[(m, j)], y[:, j], w) for j in organs]) for m in models}
                for a, c in PAIRS:
                    diffs[(a, c)].append(auc[a] - auc[c])
            for pair in PAIRS:
                draws[pair][b] = np.mean(diffs[pair])
        for (a, c), v in draws.items():
            lo, hi = np.percentile(v[1:], [2.5, 97.5])
            rows.append({"split": split, "a": a, "b": c, "mean_diff": round(v[0], 4), "boot_ci95_low": round(lo, 4),
                         "boot_ci95_high": round(hi, 4), "p_boot_a_better": round((v[1:] > 0).mean(), 3), "B": B})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "bootstrap.csv", index=False)


def _self_check():
    r = np.random.default_rng(1)
    s, y, w = np.round(r.random(500), 2), (r.random(500) < 0.3).astype(float), r.integers(0, 4, 500).astype(float)
    u, inv = np.unique(s, return_inverse=True)
    assert abs(weighted_auc(inv.ravel(), len(u), y, w) - roc_auc_score(y, s, sample_weight=w)) < 1e-12


if __name__ == "__main__":
    _self_check()
    main()
