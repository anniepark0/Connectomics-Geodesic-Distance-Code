"""
Interactive 3D viewer for PAM-alpha1 (PAM11) using REAL triangulated mesh
surfaces (true soma shape included) + postsynaptic sites.

Reads the CSVs from pam_alpha1_figure_mesh.py - no network needed, runs in
seconds, safe to re-run while tweaking colors below.

Run with: python3 pam_alpha1_figure_3d_mesh.py
Then open PAM_alpha1_inputs_3D.html in a browser.

Controls:
- Drag to rotate, scroll to zoom, double-click to reset view.
- Legend (top-right): click a neuron to hide/show its mesh + synapses together
  (this also covers hiding all synapses for a neuron, since they share a toggle).
- Dropdown (bottom-left): mesh color presets (soma stays a separate, fixed
  highlight color so the cell body remains visually distinct - edit
  SOMA_COLOR below to change it).
- Slider (bottom-center): synapse marker size.
- Slider (bottom-right): mesh opacity.
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import plotly.graph_objects as go

# ── CONFIG - edit these and re-run any time, no network needed ──────────
MESH_COLOR_MODE = "density"  # "flat" (original solid color) or "density"
                              # (heatmap showing local synapse density -
                              # sparse/empty regions like a soma-adjacent
                              # "synapse desert" show up as a distinct color)
DENSITY_BANDWIDTH = 2000     # nm - how 'local' the density estimate is.
                              # Smaller = sharper/noisier, larger = smoother.
                              # Try values comparable to typical synapse
                              # spacing in your data.
DENSITY_COLORSCALE = "Viridis"  # try "Hot", "Plasma", "Turbo" for other looks
MESH_COLOR = "forestgreen"   # only used when MESH_COLOR_MODE = "flat"
MESH_OPACITY = 1.0
SOMA_COLOR = "darkgreen"     # only used when MESH_COLOR_MODE = "flat"
SOMA_OPACITY = 1.0
SYNAPSE_COLOR = "#F2C00C"
SYNAPSE_SIZE = 3
SYNAPSE_OPACITY = 0.85
BACKGROUND_COLOR = "white"
OUTPUT_HTML = "PAM_alpha1_inputs_3D.html"
OUTPUT_PNG = "PAM_alpha1_inputs_3D.png"
PNG_WIDTH, PNG_HEIGHT = 1600, 1600
CAMERA = dict(eye=dict(x=1.5, y=1.5, z=1.2))
# ──────────────────────────────────────────────────────────────────────


def compute_vertex_density(vertex_xyz, synapse_xyz, bandwidth):
    """Per-vertex synapse density via a Gaussian kernel density estimate:
    each vertex's value is the sum of Gaussian-weighted contributions from
    every synapse within ~3 bandwidths. Regions with no nearby synapses
    (a 'synapse desert') come out at or near zero."""
    if len(synapse_xyz) == 0:
        return np.zeros(len(vertex_xyz))
    tree = cKDTree(synapse_xyz)
    density = np.zeros(len(vertex_xyz))
    pairs = tree.query_ball_point(vertex_xyz, r=bandwidth * 3)
    for i, neighbors in enumerate(pairs):
        if not neighbors:
            continue
        dists = np.linalg.norm(synapse_xyz[neighbors] - vertex_xyz[i], axis=1)
        density[i] = np.sum(np.exp(-0.5 * (dists / bandwidth) ** 2))
    return density

vertices = pd.read_csv("pam_alpha1_mesh_vertices.csv", dtype={"neuron_id": str})
faces = pd.read_csv("pam_alpha1_mesh_faces.csv", dtype={"neuron_id": str})
synapses = pd.read_csv("pam_alpha1_synapses.csv", dtype={"neuron_id": str})

neuron_ids = sorted(vertices["neuron_id"].unique())
print(f"{len(neuron_ids)} neurons found. Legend mapping:")
for i, nid in enumerate(neuron_ids, 1):
    print(f"  Neuron {i}  =  root ID {nid}")

traces = []
mesh_trace_idx, soma_trace_idx, synapse_trace_idx = [], [], []

for i, nid in enumerate(neuron_ids, 1):
    label = f"Neuron {i}"

    v = vertices[vertices["neuron_id"] == nid].sort_values("vertex_index")
    f = faces[faces["neuron_id"] == nid]
    ssub = synapses[synapses["neuron_id"] == nid]

    if MESH_COLOR_MODE == "density":
        vertex_xyz = v[["x", "y", "z"]].values
        syn_xyz = ssub[["x", "y", "z"]].values if len(ssub) else np.empty((0, 3))
        density = compute_vertex_density(vertex_xyz, syn_xyz, DENSITY_BANDWIDTH)
        common_kwargs = dict(
            intensity=density, colorscale=DENSITY_COLORSCALE,
            cmin=0, cmax=max(density.max(), 1e-9),
            showscale=(i == 1),
            colorbar=dict(title="Synapse<br>density") if i == 1 else None,
        )
    else:
        common_kwargs = dict(color=MESH_COLOR, opacity=MESH_OPACITY)

    # ── full mesh (every vertex/face for this neuron - includes the soma
    # naturally, since it's part of the same continuous reconstructed surface) ──
    traces.append(go.Mesh3d(
        x=v["x"], y=v["y"], z=v["z"],
        i=f["i"], j=f["j"], k=f["k"],
        flatshading=False,
        name=label, legendgroup=nid, showlegend=True,
        hoverinfo="skip",
        **common_kwargs,
    ))
    mesh_trace_idx.append(len(traces) - 1)

    # ── soma overlay: same real geometry, restricted to detected soma
    # vertices/faces. In density mode it shares the SAME density array/scale
    # as the main mesh (so the gradient is continuous across the soma, not
    # masked by a flat highlight color) - in flat mode it's a distinct
    # highlight color as before. ──
    soma_v_idx = set(v.loc[v["is_soma"], "vertex_index"])
    soma_face_mask = f.apply(
        lambda row: row["i"] in soma_v_idx and row["j"] in soma_v_idx and row["k"] in soma_v_idx,
        axis=1,
    )
    f_soma = f[soma_face_mask]
    soma_kwargs = common_kwargs if MESH_COLOR_MODE == "density" else dict(
        color=SOMA_COLOR, opacity=SOMA_OPACITY
    )
    if MESH_COLOR_MODE == "density":
        soma_kwargs = {**common_kwargs, "showscale": False}  # avoid a 2nd colorbar
    traces.append(go.Mesh3d(
        x=v["x"], y=v["y"], z=v["z"],
        i=f_soma["i"], j=f_soma["j"], k=f_soma["k"],
        flatshading=False,
        name=f"{label} soma", legendgroup=nid, showlegend=False,
        hoverinfo="skip",
        **soma_kwargs,
    ))
    soma_trace_idx.append(len(traces) - 1)

    # ── this neuron's postsynaptic sites ──
    traces.append(go.Scatter3d(
        x=ssub["x"], y=ssub["y"], z=ssub["z"], mode="markers",
        marker=dict(size=SYNAPSE_SIZE, color=SYNAPSE_COLOR, opacity=SYNAPSE_OPACITY),
        name=f"{label} synapses", legendgroup=nid, showlegend=False,
        hoverinfo="skip",
    ))
    synapse_trace_idx.append(len(traces) - 1)

fig = go.Figure(data=traces)

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        aspectmode="data", bgcolor=BACKGROUND_COLOR,
    ),
    paper_bgcolor=BACKGROUND_COLOR,
    legend=dict(x=0.75, y=0.98, groupclick="togglegroup", title=dict(text="Neurons (click to toggle)")),
    margin=dict(l=0, r=0, t=30, b=0),
    title="PAM-α1 (PAM11) - reconstructed mesh with postsynaptic sites",
)

# ── Mesh color presets (only meaningful in "flat" mode - a solid `color`
# value has no visible effect when `intensity`-based density coloring is
# active, so skip this control entirely in "density" mode to avoid a
# dropdown that silently does nothing) ──
updatemenus = []
if MESH_COLOR_MODE == "flat":
    color_presets = {"Green": "forestgreen", "Gray": "dimgray", "Blue": "steelblue", "Black": "black"}
    color_buttons = [
        dict(label=label, method="restyle", args=[{"color": mcol}, mesh_trace_idx])
        for label, mcol in color_presets.items()
    ]
    updatemenus.append(
        dict(type="dropdown", buttons=color_buttons, x=0.0, y=0.02, xanchor="left", yanchor="bottom")
    )

size_options = [1, 2, 3, 5, 8, 12]
size_steps = [
    dict(label=str(s), method="restyle", args=[{"marker.size": s}, synapse_trace_idx])
    for s in size_options
]

# Opacity applies to BOTH the main mesh and the soma overlay together, so
# they stay visually consistent (this matters especially in density mode,
# where the two traces share one continuous color gradient).
opacity_options = [0.2, 0.4, 0.6, 0.8, 1.0]
opacity_steps = [
    dict(label=str(o), method="restyle", args=[{"opacity": o}, mesh_trace_idx + soma_trace_idx])
    for o in opacity_options
]

fig.update_layout(
    updatemenus=updatemenus,
    sliders=[
        dict(
            active=size_options.index(SYNAPSE_SIZE) if SYNAPSE_SIZE in size_options else 2,
            currentvalue={"prefix": "Synapse size: "},
            pad={"t": 50},
            len=0.4, x=0.3, y=0.02,
            steps=size_steps,
        ),
        dict(
            active=opacity_options.index(MESH_OPACITY) if MESH_OPACITY in opacity_options else 4,
            currentvalue={"prefix": "Mesh opacity: "},
            pad={"t": 50},
            len=0.4, x=0.75, y=0.02,
            steps=opacity_steps,
        ),
    ],
)

fig.write_html(
    OUTPUT_HTML, include_plotlyjs="cdn",
    # Removes the broken "download as png" toolbar button - it can't reliably
    # capture WebGL (3D) content (a known Plotly limitation: the browser
    # clears the WebGL buffer before the screenshot is taken), so leaving it
    # in place just produces the toolbar-but-no-mesh result you saw. Use the
    # write_image() export below instead for a clean static PNG.
    config={"modeBarButtonsToRemove": ["toImage"]},
)
print(f"Saved {OUTPUT_HTML} - open it in a browser to explore")

# ── Clean static PNG export (no toolbar/buttons, since this is a direct
# headless render, not a browser screenshot) ──
# Requires: pip install kaleido
# If you get "Kaleido requires Google Chrome to be installed", run this once
# in your terminal: plotly_get_chrome
try:
    fig.update_layout(scene_camera=CAMERA)
    fig.write_image(OUTPUT_PNG, width=PNG_WIDTH, height=PNG_HEIGHT)
    print(f"Saved {OUTPUT_PNG}")
except Exception as e:
    print(f"PNG export skipped ({type(e).__name__}: {e}). "
          f"If this mentions Chrome, run `plotly_get_chrome` in your terminal "
          f"and re-run this script.")
