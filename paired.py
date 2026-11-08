"""Paired per-seed differences for the comparisons reported in the paper."""
import pandas as pd
from scipy import stats

m = pd.read_csv("results/metrics.csv")
pairs = [("temporal", "drug_id_patient", "drug_frequency"), ("temporal", "drug_id_patient", "graph_patient"),
         ("temporal", "graph_patient", "graph"), ("cold_drug", "graph_patient", "patient_only"),
         ("cold_drug", "graph_patient", "graph"), ("temporal", "drug_id_patient", "gradient_boosting"),
         ("cold_drug", "graph_patient", "gradient_boosting"), ("cold_drug", "graph_patient", "logistic_regression"),
         ("cold_drug", "graph_patient", "structure_knn")]
rows = []
for split, a, b in pairs:
    w = m[m.split == split].pivot(index="seed", columns="model", values="auroc")
    d = w[a] - w[b]
    half = stats.t.ppf(0.975, len(d) - 1) * d.std() / len(d) ** 0.5
    rows.append({"split": split, "a": a, "b": b, "mean_diff": round(d.mean(), 4), "sd": round(d.std(), 4),
                 "ci95_low": round(d.mean() - half, 4), "ci95_high": round(d.mean() + half, 4),
                 "seeds_a_wins": int((d > 0).sum()), "n": len(d)})
out = pd.DataFrame(rows)
out.to_csv("results/paired.csv", index=False)
print(out.to_string(index=False))
