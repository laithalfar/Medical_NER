import pandas as pd
import json
import sys
import re
import os


# 1. Load your CSV
df = pd.read_csv("imports/MESH.csv")

# 2. Adjust these names to match your actual columns:
MESH_COL = "Class ID"      # e.g. "DescriptorUI", "MESH_ID", ...
UMLS_COL = "CUI"     # e.g. "CUI", "UMLS_CUI", ...

# 3. Drop rows with missing mappings
df = df[[ MESH_COL, UMLS_COL]].dropna()

df[MESH_COL] = [url.rstrip("/").split("/")[-1]  for url in df[MESH_COL]]   
#.astype(str)

# 4. Build dictionary: {mesh_id: umls_cui}
mesh_to_umls = {
    str(row[MESH_COL]).strip(): str(row[UMLS_COL]).strip()
    for _, row in df.iterrows()
}

# 5. Save as JSON
with open("data/mesh_umls_map.json", "w", encoding="utf-8") as f:
    json.dump(mesh_to_umls, f, ensure_ascii=False, indent=2)

print(f"Saved {len(mesh_to_umls)} MeSH→UMLS mappings to mesh_umls_map.json")
