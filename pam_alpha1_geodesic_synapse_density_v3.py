#!/usr/bin/env python3
"""
Publication-style quantification of input synapse density along PAM-alpha1 neurons.

Reads the CSVs produced by pam_alpha1_flattened_fetch.py:
    pam_alpha1_flat_nodes.csv
    pam_alpha1_flat_synapses.csv
    pam_alpha1_flat_soma.csv

Outputs:
    PAM_alpha1_synapse_density_per_neuron.pdf/svg
        Each neuron: skeleton coloured by local geodesic input density +
        density-vs-cable-distance curve and synapse rug.

    PAM_alpha1_synapse_density_population.pdf/svg
        Population heatmap after normalising soma-rooted cable distance to [0, 1],
        plus mean +/- 95% CI density profile.

    PAM_alpha1_synapse_density_summary.csv
        Per-neuron summary statistics.

    PAM_alpha1_synapse_density_nodes.csv
        Density value assigned to every skeleton node.

Method:
  * Synapses are assigned to their nearest skeleton node in 3D.
  * Distances are measured along the skeleton (geodesic/cable distance).
  * Local density is a Gaussian kernel over geodesic distance.
  * Density is normalised by sqrt(2*pi)*bandwidth, giving synapses/nm,
    and plotted as synapses per 10 micrometres.
  * The soma-nearest skeleton node is used as the distance origin.

Requires:
    pip install pandas numpy networkx scipy matplotlib
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d

# ---------------------------- CONFIG ---------------------------------
NODES_CSV = "pam_alpha1_flat_nodes.csv"
SYNAPSES_CSV = "pam_alpha1_flat_synapses.csv"
SOMAS_CSV = "pam_alpha1_flat_soma.csv"

# Gaussian bandwidth along the skeleton. 10 um is a sensible starting point;
# re-run at e.g. 5, 10 and 20 um as a robustness check for a paper.
BANDWIDTH_UM = 10.0

# Sampling resolution for 1-D profiles.
BIN_UM = 2.0

# Figures
NCOLS = 2
CMAP = "inferno"
SKELETON_LINEWIDTH = 1.2
RAW_SKELETON_LINEWIDTH = 0.45
SYNAPSE_MARKER_SIZE = 8
SHOW_SYNAPSE_DOTS_ON_MORPHOLOGY = True

OUT_PER_NEURON = "PAM_alpha1_synapse_density_per_neuron"
OUT_POPULATION = "PAM_alpha1_synapse_density_population"
# ---------------------------------------------------------------------


def build_graph(sub):
    """Build weighted skeleton graph; coordinates are assumed to be nm."""
    G = nx.Graph()
    coords = {}
    for _, r in sub.iterrows():
        n = int(r["node_id"])
        coords[n] = np.array([r["x"], r["y"], r["z"]], dtype=float)
        G.add_node(n)
    for _, r in sub.iterrows():
        n, p = int(r["node_id"]), int(r["parent_id"])
        if p == -1 or p not in coords:
            continue
        length = float(np.linalg.norm(coords[n] - coords[p]))
        G.add_edge(n, p, weight=length)
    return G, coords


def largest_component(G):
    """FlyWire skeletons should be connected; protect against tiny fragments."""
    if nx.is_connected(G):
        return G
    cc = max(nx.connected_components(G), key=len)
    print(f"  WARNING: disconnected skeleton; using largest component "
          f"({len(cc)}/{len(G)} nodes)")
    return G.subgraph(cc).copy()


def nearest_node(point_xyz, node_ids, node_xyz):
    tree = cKDTree(node_xyz)
    _, ix = tree.query(np.asarray(point_xyz, dtype=float))
    return int(node_ids[ix])


def assign_synapses_to_nodes(ssub, node_ids, node_xyz):
    if len(ssub) == 0:
        return np.array([], dtype=int)
    tree = cKDTree(node_xyz)
    _, ix = tree.query(ssub[["x", "y", "z"]].to_numpy(float))
    return node_ids[ix].astype(int)


def soma_root(G, coords, soma_row):
    """Choose skeleton node closest to soma coordinate; fall back to graph node."""
    ids = np.array(list(G.nodes()), dtype=int)
    xyz = np.array([coords[n] for n in ids])
    if soma_row is not None:
        p = soma_row[["x", "y", "z"]].to_numpy(dtype=float)
        return nearest_node(p, ids, xyz)
    # Fallback: use a terminal nearest one end of the graph diameter.
    start = ids[0]
    d = nx.single_source_dijkstra_path_length(G, start, weight="weight")
    return int(max(d, key=d.get))


def geodesic_density(G, syn_nodes, bandwidth_nm):
    """
    Gaussian KDE evaluated at every skeleton node using shortest-path distance.
    Returns density in synapses/nm.

    This direct implementation is transparent and publication-friendly.
    For very large skeletons it can be accelerated, but PAM-alpha1 should be fine.
    """
    nodes = list(G.nodes())
    density = dict.fromkeys(nodes, 0.0)
    if len(syn_nodes) == 0:
        return density

    # Multiple synapses can map to one skeleton node: collapse and weight them.
    unique, counts = np.unique(syn_nodes, return_counts=True)
    cutoff = 3.0 * bandwidth_nm
    norm = np.sqrt(2.0 * np.pi) * bandwidth_nm

    for source, count in zip(unique, counts):
        # Only distances within 3 sigma matter appreciably.
        d = nx.single_source_dijkstra_path_length(
            G, int(source), cutoff=cutoff, weight="weight"
        )
        for node, dist in d.items():
            density[node] += count * np.exp(-0.5 * (dist / bandwidth_nm) ** 2) / norm
    return density


def cable_distance_profile(G, root, syn_nodes, bin_nm, bandwidth_nm):
    """
    1-D soma-rooted profile. Note that distance-from-soma collapses different
    branches onto one axis, so use this together with the morphology heatmap.
    """
    dist = nx.single_source_dijkstra_path_length(G, root, weight="weight")
    syn_d = np.array([dist[int(n)] for n in syn_nodes if int(n) in dist], dtype=float)

    max_d = max(dist.values()) if dist else 0.0
    edges = np.arange(0, max_d + bin_nm * 1.5, bin_nm)
    if len(edges) < 2:
        edges = np.array([0.0, bin_nm])
    centers = (edges[:-1] + edges[1:]) / 2

    counts, _ = np.histogram(syn_d, bins=edges)
    # Count profile smoothed with Gaussian sigma = bandwidth/bin width.
    sigma_bins = bandwidth_nm / bin_nm
    smooth_counts = gaussian_filter1d(counts.astype(float), sigma=sigma_bins,
                                      mode="constant")
    # Convert to synapses per 10 um.
    density_per_10um = smooth_counts / (bin_nm / 10000.0)
    return centers, density_per_10um, syn_d, dist


def morphology_segments(G, coords, density):
    segs, vals = [], []
    for a, b in G.edges():
        segs.append([coords[a] / 1000.0, coords[b] / 1000.0])  # um for plotting
        vals.append((density[a] + density[b]) / 2.0 * 10000.0)
    return np.asarray(segs), np.asarray(vals)


def _column_nanmean_no_warning(a):
    """Column-wise NaN mean without warnings for columns containing only NaNs."""
    valid_n = np.sum(np.isfinite(a), axis=0)
    sums = np.nansum(a, axis=0)
    out = np.full(a.shape[1], np.nan, dtype=float)
    np.divide(sums, valid_n, out=out, where=valid_n > 0)
    return out


def _column_nanpercentile_no_warning(a, q):
    """Column-wise percentile, leaving all-NaN columns as NaN without warnings."""
    out = np.full(a.shape[1], np.nan, dtype=float)
    for j in range(a.shape[1]):
        vals = a[:, j]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[j] = np.percentile(vals, q)
    return out


def bootstrap_ci(matrix, n_boot=2000, seed=7):
    """Bootstrap neurons (rows), returning mean and 95% CI.

    Columns with no contributing neurons remain NaN. This is expected at
    actual cable distances beyond the ends of all neurons in a resample and
    is handled explicitly so NumPy does not emit 'empty slice' warnings.
    """
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    boots = np.full((n_boot, matrix.shape[1]), np.nan, dtype=float)

    for i in range(n_boot):
        ix = rng.integers(0, n, size=n)
        boots[i] = _column_nanmean_no_warning(matrix[ix])

    mean = _column_nanmean_no_warning(matrix)
    lo = _column_nanpercentile_no_warning(boots, 2.5)
    hi = _column_nanpercentile_no_warning(boots, 97.5)
    return mean, lo, hi


nodes = pd.read_csv(NODES_CSV, dtype={"neuron_id": str})
synapses = pd.read_csv(SYNAPSES_CSV, dtype={"neuron_id": str})
somas = pd.read_csv(SOMAS_CSV, dtype={"neuron_id": str})

neuron_ids = sorted(nodes["neuron_id"].unique())
bw_nm = BANDWIDTH_UM * 1000.0
bin_nm = BIN_UM * 1000.0

results = {}
summary_rows = []
node_density_rows = []

print(f"Analysing {len(neuron_ids)} neurons")
print(f"Geodesic KDE bandwidth = {BANDWIDTH_UM:g} um")

for i, nid in enumerate(neuron_ids, 1):
    print(f"[{i}/{len(neuron_ids)}] {nid}")
    sub = nodes[nodes["neuron_id"] == nid].copy()
    ssub = synapses[synapses["neuron_id"] == nid].copy()
    ssoma = somas[somas["neuron_id"] == nid]

    G0, coords0 = build_graph(sub)
    G = largest_component(G0)
    coords = {n: coords0[n] for n in G.nodes()}

    node_ids = np.array(list(G.nodes()), dtype=int)
    node_xyz = np.array([coords[n] for n in node_ids])
    syn_nodes = assign_synapses_to_nodes(ssub, node_ids, node_xyz)

    soma_row = ssoma.iloc[0] if len(ssoma) else None
    root = soma_root(G, coords, soma_row)

    dens = geodesic_density(G, syn_nodes, bw_nm)
    centers, profile, syn_d, root_dist = cable_distance_profile(
        G, root, syn_nodes, bin_nm, bw_nm
    )

    segs, seg_vals = morphology_segments(G, coords, dens)
    total_length_nm = sum(d["weight"] for _, _, d in G.edges(data=True))
    max_root_dist_nm = max(root_dist.values())

    for n in G.nodes():
        node_density_rows.append({
            "neuron_id": nid,
            "node_id": n,
            "distance_from_soma_um": root_dist.get(n, np.nan) / 1000.0,
            "input_density_per_10um": dens[n] * 10000.0,
        })

    summary_rows.append({
        "neuron_id": nid,
        "n_input_synapses": len(syn_nodes),
        "total_cable_length_um": total_length_nm / 1000.0,
        "global_inputs_per_10um": len(syn_nodes) / (total_length_nm / 10000.0),
        "max_soma_rooted_distance_um": max_root_dist_nm / 1000.0,
        "peak_profile_density_per_10um": float(np.max(profile)) if len(profile) else np.nan,
        "peak_profile_distance_um": float(centers[np.argmax(profile)] / 1000.0)
            if len(profile) else np.nan,
    })

    results[nid] = dict(
        G=G, coords=coords, syn_nodes=syn_nodes, root=root,
        density=dens, centers=centers, profile=profile, syn_d=syn_d,
        segs=segs, seg_vals=seg_vals, node_ids=node_ids, node_xyz=node_xyz
    )

pd.DataFrame(summary_rows).to_csv("PAM_alpha1_synapse_density_summary.csv", index=False)
pd.DataFrame(node_density_rows).to_csv("PAM_alpha1_synapse_density_nodes.csv", index=False)


# ==================== FIGURE 1: EACH NEURON ============================
nrows = len(neuron_ids)
fig = plt.figure(figsize=(10.5, max(3.0 * nrows, 6)))
gs = fig.add_gridspec(nrows, 2, width_ratios=[1.15, 1.0], hspace=0.38, wspace=0.28)

# Use one common density scale so colour is quantitatively comparable.
all_seg_vals = np.concatenate([results[n]["seg_vals"] for n in neuron_ids])
vmax = np.nanpercentile(all_seg_vals, 99) if len(all_seg_vals) else 1.0
vmax = max(vmax, 1e-9)

last_lc = None
for row, nid in enumerate(neuron_ids):
    r = results[nid]

    # A: morphology coloured by local geodesic density
    ax = fig.add_subplot(gs[row, 0], projection="3d")
    lc = LineCollection([])  # placeholder only; 3-D lines are drawn edge-wise
    norm = plt.Normalize(0, vmax)
    cmap = plt.get_cmap(CMAP)

    for (a, b), val in zip(r["G"].edges(), r["seg_vals"]):
        pa, pb = r["coords"][a] / 1000.0, r["coords"][b] / 1000.0
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                color=cmap(norm(val)), lw=SKELETON_LINEWIDTH,
                solid_capstyle="round")

    if SHOW_SYNAPSE_DOTS_ON_MORPHOLOGY and len(r["syn_nodes"]):
        syn_xyz = np.array([r["coords"][int(n)] for n in r["syn_nodes"]]) / 1000.0
        ax.scatter(syn_xyz[:, 0], syn_xyz[:, 1], syn_xyz[:, 2],
                   s=SYNAPSE_MARKER_SIZE, facecolors="none", edgecolors="black",
                   linewidths=0.35, alpha=0.65)

    root_xyz = r["coords"][r["root"]] / 1000.0
    ax.scatter(*root_xyz, s=28, marker="o", facecolor="white",
               edgecolor="black", linewidth=0.8, zorder=10)
    ax.set_axis_off()
    ax.set_title(f"Neuron {row + 1}  |  n={len(r['syn_nodes'])} inputs",
                 fontsize=9, pad=0)

    # B: soma-rooted density curve + raw synapse rug
    ax2 = fig.add_subplot(gs[row, 1])
    x_um = r["centers"] / 1000.0
    ax2.plot(x_um, r["profile"], lw=1.6)
    if len(r["syn_d"]):
        rug_y = -0.055 * max(np.max(r["profile"]), 1)
        ax2.scatter(r["syn_d"] / 1000.0,
                    np.full(len(r["syn_d"]), rug_y),
                    marker="|", s=24, linewidths=0.45, clip_on=False)
    ax2.axhline(0, lw=0.6)
    ax2.set_ylabel("Inputs / 10 µm", fontsize=8)
    if row == nrows - 1:
        ax2.set_xlabel("Cable distance from soma (µm)", fontsize=8)
    else:
        ax2.set_xticklabels([])
    ax2.tick_params(labelsize=7)
    ax2.spines[["top", "right"]].set_visible(False)

# Shared colour bar via ScalarMappable.
sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap=CMAP)
sm.set_array([])
cbar_ax = fig.add_axes([0.08, 0.035, 0.36, 0.012])
cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cb.set_label("Local geodesic input density (synapses / 10 µm)", fontsize=8)
cb.ax.tick_params(labelsize=7)

fig.suptitle(
    f"PAM-α1 input synapse density along the neuronal arbor "
    f"(Gaussian geodesic KDE, σ={BANDWIDTH_UM:g} µm)",
    fontsize=11, y=0.997
)
fig.subplots_adjust(left=0.04, right=0.98, top=0.975, bottom=0.07)
fig.savefig(OUT_PER_NEURON + ".pdf", dpi=300)
fig.savefig(OUT_PER_NEURON + ".svg")
plt.close(fig)


# ==================== FIGURE 2: POPULATION =============================
# Interpolate each neuron onto normalised soma-rooted cable distance [0, 1].
grid = np.linspace(0, 1, 101)
mat = []
for nid in neuron_ids:
    r = results[nid]
    x = r["centers"]
    if len(x) < 2 or x[-1] <= 0:
        mat.append(np.full_like(grid, np.nan))
        continue
    xn = x / x[-1]
    mat.append(np.interp(grid, xn, r["profile"], left=np.nan, right=np.nan))
mat = np.asarray(mat)

mean, lo, hi = bootstrap_ci(mat)

fig = plt.figure(figsize=(8.2, 6.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1], hspace=0.42)

ax = fig.add_subplot(gs[0])
im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=CMAP,
               extent=[0, 1, len(neuron_ids) + 0.5, 0.5])
ax.set_ylabel("Neuron", fontsize=9)
ax.set_yticks(np.arange(1, len(neuron_ids) + 1))
ax.set_yticklabels([str(i) for i in range(1, len(neuron_ids) + 1)], fontsize=7)
ax.set_xlabel("Normalised soma-rooted cable distance", fontsize=9)
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("Inputs / 10 µm", fontsize=8)

ax2 = fig.add_subplot(gs[1])
ax2.plot(grid, mean, lw=2, label="Mean")
ax2.fill_between(grid, lo, hi, alpha=0.22, linewidth=0, label="95% bootstrap CI")
for row in mat:
    ax2.plot(grid, row, lw=0.45, alpha=0.22)
ax2.set_xlabel("Normalised soma-rooted cable distance", fontsize=9)
ax2.set_ylabel("Inputs / 10 µm", fontsize=9)
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(frameon=False, fontsize=8)

fig.suptitle("Population distribution of PAM-α1 input synapses", fontsize=11)
fig.savefig(OUT_POPULATION + ".pdf", bbox_inches="tight", dpi=300)
fig.savefig(OUT_POPULATION + ".svg", bbox_inches="tight")
plt.close(fig)


# ================= FIGURE 3: ACTUAL-DISTANCE POPULATION ===============
# Actual (not normalized) soma-rooted cable distance.
# Top panel: one row per neuron; every blue vertical hatch is one actual
# postsynaptic input synapse. No heatmap is used.
# Bottom panel: density profiles aligned in actual micrometres.

OUT_ACTUAL_DISTANCE = "PAM_alpha1_synapse_density_population_actual_distance"

fig = plt.figure(figsize=(8.2, 6.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1.0], hspace=0.42)

ax = fig.add_subplot(gs[0])
for i, nid in enumerate(neuron_ids, 1):
    syn_um = results[nid]["syn_d"] / 1000.0
    if len(syn_um):
        ax.vlines(
            syn_um, i - 0.34, i + 0.34,
            color="tab:blue", linewidth=0.45, alpha=0.75
        )

ax.set_facecolor("white")
# Reference distances
REFERENCE_DISTANCES_UM = [72.01, 154.2, 160.0]

for x in REFERENCE_DISTANCES_UM:
    ax.axvline(
        x,
        color="black",
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
        zorder=0
    )
ax.set_ylabel("Neuron", fontsize=9)
ax.set_yticks(np.arange(1, len(neuron_ids) + 1))
ax.set_yticklabels([str(i) for i in range(1, len(neuron_ids) + 1)], fontsize=7)
ax.set_ylim(len(neuron_ids) + 0.5, 0.5)
ax.set_xlim(left=0)
ax.set_xlabel("Cable distance from soma (µm)", fontsize=9)
ax.tick_params(labelsize=7)
ax.spines[["top", "right"]].set_visible(False)

# Common actual-distance grid. NaN beyond each neuron's observed arbor means
# short neurons do not contribute artificial zero-density values at long range.
max_distance_um = max(
    (results[nid]["centers"][-1] / 1000.0
     for nid in neuron_ids if len(results[nid]["centers"])),
    default=1.0
)
actual_grid_um = np.arange(0, max_distance_um + BIN_UM, BIN_UM)

actual_mat = []
for nid in neuron_ids:
    r = results[nid]
    x_um = r["centers"] / 1000.0
    y = r["profile"]

    yi = np.full(actual_grid_um.shape, np.nan, dtype=float)
    if len(x_um) >= 2:
        valid = (actual_grid_um >= x_um[0]) & (actual_grid_um <= x_um[-1])
        yi[valid] = np.interp(actual_grid_um[valid], x_um, y)
    actual_mat.append(yi)

actual_mat = np.asarray(actual_mat)
actual_mean, actual_lo, actual_hi = bootstrap_ci(actual_mat)

ax2 = fig.add_subplot(gs[1], sharex=ax)

# Same anatomical reference distances on the lower density panel.
for x in REFERENCE_DISTANCES_UM:
    ax2.axvline(
        x,
        color="black",
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
        zorder=0
    )

for row in actual_mat:
    ax2.plot(actual_grid_um, row, lw=0.45, alpha=0.22, color="tab:blue")

ax2.plot(actual_grid_um, actual_mean, lw=2.0,
         color="tab:blue", label="Mean")
ax2.fill_between(
    actual_grid_um, actual_lo, actual_hi,
    color="tab:blue", alpha=0.22, linewidth=0,
    label="95% bootstrap CI"
)

ax2.set_xlabel("Cable distance from soma (µm)", fontsize=9)
ax2.set_ylabel("Inputs / 10 µm", fontsize=9)
ax2.tick_params(labelsize=7)
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(frameon=False, fontsize=8)

fig.suptitle(
    "Population distribution of PAM-α1 input synapses — actual cable distance",
    fontsize=11
)
fig.savefig(OUT_ACTUAL_DISTANCE + ".pdf", bbox_inches="tight", dpi=300)
fig.savefig(OUT_ACTUAL_DISTANCE + ".svg", bbox_inches="tight")
plt.close(fig)

print(f"  {OUT_ACTUAL_DISTANCE}.pdf/.svg")
# =====================================================================


print("\nDone.")
print(f"  {OUT_PER_NEURON}.pdf/.svg")
print(f"  {OUT_POPULATION}.pdf/.svg")
print("  PAM_alpha1_synapse_density_summary.csv")
print("  PAM_alpha1_synapse_density_nodes.csv")
