"""
Fetch skeletons + filtered postsynaptic sites for PAM-alpha1 (PAM11), for the
flattened/unrolled maps (plain or density-heatmap version). Uses skeletons
(not full meshes), since the unrolling is based on skeleton topology, not
mesh surface detail.

Run with: python3 pam_alpha1_flattened_fetch.py
Requires: pip install fafbseg navis pandas numpy
"""

import pandas as pd
from fafbseg import flywire

flywire.set_default_dataset("public")

# ── 1. Get PAM11 (PAM-alpha1) root IDs ──
annotations = pd.read_csv(
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv",
    sep="\t", dtype={"root_id": str, "supervoxel_id": str}
)
pam_a1 = annotations[annotations["cell_type"] == "PAM11"]
rootids = pam_a1["root_id"].tolist()
print(f"Found {len(pam_a1)} PAM11 neurons")

# ── 2. Fetch skeletons (static file server, materialization 783 - matches ──
# these root IDs directly, no production access needed)
skeletons = flywire.get_skeletons(rootids, dataset=783, progress=True)
print(skeletons)

node_rows = []
for n in skeletons:
    nid_neuron = str(n.id)
    for _, row in n.nodes.iterrows():
        node_rows.append({
            "neuron_id": nid_neuron,
            "node_id": int(row["node_id"]),
            "parent_id": int(row["parent_id"]),
            "x": float(row["x"]), "y": float(row["y"]), "z": float(row["z"]),
        })
pd.DataFrame(node_rows).to_csv("pam_alpha1_flat_nodes.csv", index=False)
print(f"Saved pam_alpha1_flat_nodes.csv ({len(node_rows)} nodes)")

# Sanity check - confirms no fetch-order mislabeling
exported_ids = {row["neuron_id"] for row in node_rows}
missing = set(rootids) - exported_ids
if missing:
    print(f"WARNING: {len(missing)} requested neuron(s) missing from export: {missing}")
else:
    print(f"Sanity check passed: all {len(rootids)} requested neurons present.")

# ── 3. Soma positions ──
# soma_x/y/z in the annotation table are in RAW VOXEL units (4x4x40nm voxels),
# but skeleton coordinates from get_skeletons() are in actual nanometres -
# convert so the soma position lines up with the skeleton.
VOXEL_DIMS = (4, 4, 40)
soma_df = pam_a1[["root_id", "soma_x", "soma_y", "soma_z"]].dropna().copy()
soma_df["neuron_id"] = soma_df["root_id"].astype(str)
soma_df["x"] = soma_df["soma_x"].astype(float) * VOXEL_DIMS[0]
soma_df["y"] = soma_df["soma_y"].astype(float) * VOXEL_DIMS[1]
soma_df["z"] = soma_df["soma_z"].astype(float) * VOXEL_DIMS[2]
soma_df[["neuron_id", "x", "y", "z"]].to_csv("pam_alpha1_flat_soma.csv", index=False)
print(f"Saved pam_alpha1_flat_soma.csv ({len(soma_df)} somas)")

# ── 4. Fetch postsynaptic sites ──
# filtered=False + min_score=51 avoids a broken server-side view while still
# matching FlyWire's own confidence threshold (cleft_score > 50).
synapses = flywire.get_synapses(
    rootids, pre=False, post=True, dataset="public",
    filtered=False, min_score=51, progress=True
)
print(f"Fetched {len(synapses)} synapses (cleft_score >= 51)")

# Filter out connections from unidentified/fragment presynaptic partners -
# only count partners that are actually recognized, classified neurons.
known_partner_ids = set(annotations["root_id"].astype(str))
before_partner_filter = len(synapses)
synapses = synapses[synapses["pre"].astype(str).isin(known_partner_ids)]
print(f"Filtered to identified/classified presynaptic partners only: "
      f"{before_partner_filter} -> {len(synapses)} synapses")

# Apply FlyWire/Codex's standard "5+ synapses per connection" threshold.
MIN_SYNAPSES_PER_CONNECTION = 5
before = len(synapses)
synapses["pair_count"] = synapses.groupby(["pre", "post"])["id"].transform("size")
synapses = synapses[synapses["pair_count"] >= MIN_SYNAPSES_PER_CONNECTION].drop(columns="pair_count")
print(f"Applied {MIN_SYNAPSES_PER_CONNECTION}+ synapses/connection threshold: "
      f"{before} -> {len(synapses)} synapses")

synapse_export = synapses[["post_x", "post_y", "post_z", "post"]].rename(
    columns={"post_x": "x", "post_y": "y", "post_z": "z", "post": "neuron_id"}
)
synapse_export["neuron_id"] = synapse_export["neuron_id"].astype(str)
synapse_export.to_csv("pam_alpha1_flat_synapses.csv", index=False)
print(f"Saved pam_alpha1_flat_synapses.csv ({len(synapse_export)} synapses)")
