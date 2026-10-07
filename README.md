**Synapse Density and Geodesic Distance Plot**
-----------------------------------------------

**Requirements**

A FlyWire account and API token, used by fafbseg to fetch meshes and synapses. 
Set the token once: 

```python
  from fafbseg import flywire
  flywire.set_chunkedgraph_secret("YOUR_TOKEN")
```


### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running

Run from the repository folder, in this order:

```bash
python3 pam_alpha1_figure_mesh.py                  # [3D mesh rendering -> output file]
python3 pam_alpha1_figure_3d_synden.py             # [3D synapse density -> output file]
python3 pam_alpha1_flattened_fetch.py              # [fetches data for the next script -> output]
python3 pam_alpha1_geodesic_synapse_density_v3.py  # [density vs geodesic distance -> output file]
```

### Data and citation

Connectome data from FlyWire (Dorkenwald et al., 2024, *Nature*;
Schlegel et al., 2024, *Nature*), with synapse predictions from
Buhmann et al., 2021, *Nature Methods*.
