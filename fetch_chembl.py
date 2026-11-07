"""Fetch approved drugs, mechanisms and target genes from the ChEMBL REST API."""
import http.client
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://www.ebi.ac.uk/chembl/api/data"
OUT = Path(__file__).parent / "data" / "processed"


def get(path, **params):
    url = f"{API}/{path}.json?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except (OSError, http.client.HTTPException):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def fetch_all(path, key, **params):
    rows, offset = [], 0
    while True:
        page = get(path, limit=1000, offset=offset, **params)
        rows += page[key]
        offset += 1000
        if not page["page_meta"]["next"]:
            return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    molecules = fetch_all("molecule", "molecules", max_phase=4, molecule_type="Small molecule")
    drugs = pd.DataFrame([{
        "chembl_id": m["molecule_chembl_id"],
        "parent_chembl_id": (m.get("molecule_hierarchy") or {}).get("parent_chembl_id") or m["molecule_chembl_id"],
        "generic_name": m["pref_name"],
        "atc": ";".join(m.get("atc_classifications") or []),
        "smiles": (m.get("molecule_structures") or {}).get("canonical_smiles"),
    } for m in molecules if m.get("pref_name")])
    drugs.to_csv(OUT / "drugs.csv", index=False)

    mechanisms = pd.DataFrame(fetch_all("mechanism", "mechanisms"))
    mechanisms = mechanisms[["molecule_chembl_id", "parent_molecule_chembl_id", "target_chembl_id",
                             "action_type", "mechanism_of_action"]].dropna(subset=["target_chembl_id"])
    mechanisms.to_csv(OUT / "mechanisms.csv", index=False)

    ids = sorted(mechanisms.target_chembl_id.unique())
    targets = []
    for i in range(0, len(ids), 20):  # large target responses get truncated by the API
        batch = ",".join(ids[i:i + 20])
        for t in fetch_all("target", "targets", target_chembl_id__in=batch):
            for c in t.get("target_components") or []:
                for s in c.get("target_component_synonyms") or []:
                    if s["syn_type"] == "GENE_SYMBOL":
                        targets.append({"target_chembl_id": t["target_chembl_id"], "target_name": t["pref_name"],
                                        "organism": t["organism"], "gene_symbol": s["component_synonym"]})
    targets = pd.DataFrame(targets).drop_duplicates()
    targets.to_csv(OUT / "targets.csv", index=False)

    print(f"drugs={len(drugs)} mechanisms={len(mechanisms)} target_genes={len(targets)}")
    linked = drugs.parent_chembl_id.isin(mechanisms.parent_molecule_chembl_id) | drugs.chembl_id.isin(mechanisms.molecule_chembl_id)
    print(f"drugs with a mechanism={int(linked.sum())}; human target rows={int((targets.organism == 'Homo sapiens').sum())}")
    assert len(drugs) > 1000 and linked.sum() > 500, "ChEMBL fetch looks incomplete"


if __name__ == "__main__":
    main()
