"""Build the FAERS patient table and the drug-gene-organ graph from real sources."""
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
QUARTERS = ["2025q1", "2025q2", "2025q3", "2025q4"]

# shortcut: keyword map stands in for licensed MedDRA SOC lookups, replace it if a MedDRA licence is available.
ORGAN_PATTERNS = {
    "Liver": r"hepat|\bliver|jaundice|cholesta|transaminase|aminotransferase|bilirubin",
    "Heart": r"cardi|myocard|arrhythm|\batrial|ventricular|\bheart|angina|electrocardiogram qt",
    "Nervous": r"headache|dizziness|seizure|convulsion|neuropath|encephal|somnolence|tremor|cerebr|paraesthesia|syncope|migraine|ataxia|\bstroke",
    "Kidney": r"\brenal|kidney|nephr|creatinine|proteinuria|dialysis",
    "Lung": r"pneumon|dyspnoea|pulmonary|\blung|respiratory|asthma|\bcough|bronch|pleural|hypoxia",
    "GI": r"nausea|vomiting|diarrhoea|constipation|abdominal|gastr|colitis|pancreat|dyspepsia|\bileus",
    "Skin": r"\brash|pruritus|urticaria|dermat|erythema|stevens-johnson|alopecia|\bskin|eczema",
    "Blood": r"anaemia|neutropenia|thrombocytopenia|leukopenia|pancytopenia|platelet|agranulocytosis",
}
ORGANS = list(ORGAN_PATTERNS)
COMPILED = {o: re.compile(p, re.I) for o, p in ORGAN_PATTERNS.items()}
SALTS = re.compile(r"\b(HYDROCHLORIDE|HCL|SODIUM|POTASSIUM|CALCIUM|MAGNESIUM|SULFATE|SULPHATE|PHOSPHATE|ACETATE|"
                   r"MALEATE|TARTRATE|BITARTRATE|FUMARATE|SUCCINATE|CITRATE|MESYLATE|BESYLATE|TOSYLATE|"
                   r"HYDROBROMIDE|NITRATE|MONOHYDRATE|DIHYDRATE|HYDRATE|ANHYDROUS)\b")


def organs_of(term):
    return [o for o, p in COMPILED.items() if p.search(str(term))]


def norm_name(name):
    name = re.sub(r"\(.*?\)", " ", str(name).upper())
    return " ".join(SALTS.sub(" ", name).split())


def read_quarter(q):
    z = zipfile.ZipFile(RAW / f"faers_ascii_{q}.zip")
    names = {n.split("/")[-1][:4]: n for n in z.namelist() if n.startswith("ASCII/") and n.endswith(".txt")}

    def rd(key, cols):
        return pd.read_csv(z.open(names[key]), sep="$", encoding="latin-1", dtype=str,
                           on_bad_lines="skip", usecols=cols)

    demo = rd("DEMO", ["primaryid", "caseid", "caseversion", "age", "age_cod", "sex", "wt", "wt_cod"])
    drug = rd("DRUG", ["primaryid", "role_cod", "prod_ai"])
    reac = rd("REAC", ["primaryid", "pt"])
    demo["quarter"] = q
    return demo, drug, reac


def age_years(df):
    per_year = {"YR": 1, "DEC": 0.1, "MON": 12, "WK": 52, "DY": 365, "HR": 8760}
    age = pd.to_numeric(df.age, errors="coerce") / df.age_cod.map(per_year)
    return age.where(age.between(0, 120))


def weight_kg(df):
    wt = pd.to_numeric(df.wt, errors="coerce") * df.wt_cod.map({"KG": 1, "LBS": 0.453592})
    return wt.where(wt.between(1, 400))


def build_patients(lookup):
    parts = [read_quarter(q) for q in QUARTERS]
    demo = pd.concat([p[0] for p in parts], ignore_index=True)
    drug = pd.concat([p[1] for p in parts], ignore_index=True)
    reac = pd.concat([p[2] for p in parts], ignore_index=True)

    # A case can be re-reported across quarters; only its latest version counts.
    demo["caseversion"] = pd.to_numeric(demo.caseversion, errors="coerce").fillna(0)
    demo = demo.sort_values(["quarter", "caseversion"]).drop_duplicates("caseid", keep="last")

    drug = drug[drug.primaryid.isin(demo.primaryid)]
    drug["drug_id"] = drug.prod_ai.map(lambda x: lookup.get(norm_name(x)))
    primary = drug[(drug.role_cod == "PS") & drug.drug_id.notna()].drop_duplicates("primaryid")
    n_other = drug[drug.role_cod != "PS"].groupby("primaryid").size().rename("n_other_drugs")

    pts = reac[reac.primaryid.isin(primary.primaryid)].dropna(subset=["pt"])
    pt_map = {pt: organs_of(pt) for pt in pts.pt.unique()}
    labels = pts.assign(organ=pts.pt.map(pt_map)).explode("organ").dropna(subset=["organ"])
    y = pd.crosstab(labels.primaryid, labels.organ).clip(upper=1).reindex(columns=ORGANS, fill_value=0)

    out = demo.merge(primary[["primaryid", "drug_id"]], on="primaryid").merge(y, left_on="primaryid", right_index=True)
    out = out.merge(n_other, left_on="primaryid", right_index=True, how="left")
    out["age_years"], out["weight_kg"] = age_years(out), weight_kg(out)
    out["n_other_drugs"] = out.n_other_drugs.fillna(0)
    out["sex"] = out.sex.where(out.sex.isin(["F", "M"]))
    unmapped = pts[~pts.pt.map(lambda p: bool(pt_map[p]))].pt.value_counts().head(15)
    print(f"reports in={len(demo)} matched primary drug={len(primary)} with organ label={len(out)}")
    print("PT rows mapped to an organ:", round(pts.pt.map(lambda p: bool(pt_map[p])).mean(), 3))
    print("top unmapped PTs:", unmapped.to_dict())
    return out[["primaryid", "quarter", "drug_id", "age_years", "sex", "weight_kg", "n_other_drugs"] + ORGANS]


def build_graph(drugs):
    mech = pd.read_csv(OUT / "mechanisms.csv")
    targets = pd.read_csv(OUT / "targets.csv")
    targets = targets[targets.organism == "Homo sapiens"]
    mech["drug_id"] = mech.parent_molecule_chembl_id.fillna(mech.molecule_chembl_id)
    drug_gene = mech.merge(targets, on="target_chembl_id")[["drug_id", "gene_symbol", "action_type"]]
    drug_gene = drug_gene[drug_gene.drug_id.isin(drugs.parent_chembl_id)].drop_duplicates()

    cols = ["GeneSymbol", "GeneID", "DiseaseName", "DiseaseID", "DirectEvidence", "OmimIDs", "PubMedIDs"]
    gd = pd.read_csv(RAW / "CTD_curated_genes_diseases.csv.gz", comment="#", names=cols, low_memory=False)
    gd["organ"] = gd.DiseaseName.map(organs_of)
    gene_organ = gd.explode("organ").dropna(subset=["organ"])[["GeneSymbol", "organ"]].drop_duplicates()
    gene_organ.columns = ["gene_symbol", "organ"]
    return drug_gene, gene_organ


def main():
    assert organs_of("Adrenal insufficiency") == [] and organs_of("Renal failure") == ["Kidney"]
    assert organs_of("Hepatic failure") == ["Liver"] and norm_name("METFORMIN HYDROCHLORIDE") == "METFORMIN"

    drugs = pd.read_csv(OUT / "drugs.csv")
    lookup = {norm_name(n): p for n, p in zip(drugs.generic_name, drugs.parent_chembl_id)}
    patients = build_patients(lookup)
    drug_gene, gene_organ = build_graph(drugs)

    patients.to_csv(OUT / "patients.csv.gz", index=False)
    drug_gene.to_csv(OUT / "drug_gene.csv", index=False)
    gene_organ.to_csv(OUT / "gene_organ.csv", index=False)

    print(f"drugs in reports={patients.drug_id.nunique()} drug-gene edges={len(drug_gene)} gene-organ edges={len(gene_organ)}")
    print("label rates:", patients[ORGANS].mean().round(3).to_dict())
    print("rows by quarter:", patients.quarter.value_counts().sort_index().to_dict())
    with_graph = patients.drug_id.isin(drug_gene.drug_id).mean()
    print("reports whose drug has a graph target:", round(with_graph, 3))


if __name__ == "__main__":
    main()
