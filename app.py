# app.py
import streamlit as st
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw

import py3Dmol
from stmol import showmol

from core import run_job, zip_all_outputs, zip_minimized_only



st.set_page_config(
    page_title="pKaNET Cloud",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      [data-testid="stSidebar"] {min-width: 320px; max-width: 360px;}
      .small-note {font-size: 0.92rem; opacity: 0.85;}
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    **pKaNET Cloud** prepares pH-dependent ligand structures for molecular docking
    and molecular dynamics simulations.

    Workflow:
    **SMILES / ligand input → pKa prediction → protonation at selected pH →
    3D structure generation & minimization → export**
    """
)

st.caption("Tip: Use **SMI_FILE** to process a list of SMILES in batch.")

st.markdown("---")

with st.expander("What does this tool do? (Workflow)", expanded=False):
    st.image("Pka_WF.png", use_column_width=True)



# -----------------------------
# UI helpers
# -----------------------------
def show_smiles_2d(smiles: str, title: str):
    st.markdown(f"**{title}**")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.warning("Cannot parse SMILES for 2D drawing.")
        return
    img = Draw.MolToImage(mol, size=(520, 320))
    st.image(img, use_column_width=False)


def show_pdb_3d(pdb_text: str):
    view = py3Dmol.view(width=720, height=480)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"stick": {}})
    view.setBackgroundColor("white")
    view.zoomTo()
    showmol(view, height=480, width=720)


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Input")

input_type = st.sidebar.radio(
    "Choose input type",
    ["SMILES", "SMI_FILE", "FILE"],
    help="SMILES: single molecule. SMI_FILE: list of SMILES. FILE: upload PDB/MOL2/SDF."
)

st.sidebar.header("Chemistry settings")
target_pH = st.sidebar.slider("Target pH", 2.0, 12.0, 7.0, 0.1)
output_format = st.sidebar.selectbox(
    "Output format",
    ["PDB", "SDF", "MOL2"],
    index=0,
    help="MOL2 may fall back to SDF depending on server support."
)

st.sidebar.header("Naming")
output_name = st.sidebar.text_input(
    "Base output name",
    value="ligand",
    help="Used for single SMILES/FILE. For SMI_FILE, names come from the file."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="small-note">'
    'Outputs include: minimized structure, a viewer PDB, and a summary file.'
    '</div>',
    unsafe_allow_html=True
)

with st.expander("How to use (examples)", expanded=False):
    st.markdown(
        """
**1) Single SMILES**
- Select `SMILES`, paste a SMILES, choose pH and output format, then Run.

**2) SMILES list (.smi)**
- Select `SMI_FILE` and upload a text file with one molecule per line:

```text
CCO ethanol
CC(=O)O acetic_acid
c1ccccc1 benzene


---

## 5) Make results section clearer (tabs)
After the job finishes, present results as tabs:

```python
tab1, tab2, tab3 = st.tabs(["Overview", "2D / 3D View", "Downloads"])

with tab1:
    st.text_area("Summary", out["summary_text"], height=220)

with tab2:
    # your selectbox + 2D + 3D viewer here

with tab3:
    # your download buttons here


# -----------------------------
# Input widgets
# -----------------------------
smiles_text = None
uploaded = None

if input_type == "SMILES":
    smiles_text = st.text_area(
        "SMILES",
        height=130,
        placeholder="Paste a SMILES here…"
    )

elif input_type == "SMI_FILE":
    uploaded = st.file_uploader(
        "Upload .smi (one SMILES per line, optional name column)",
        type=["smi", "txt"]
    )

else:
    uploaded = st.file_uploader(
        "Upload ligand file",
        type=["pdb", "mol2", "sdf"]
    )


# Optional: protect your server
MAX_MB = 4
if uploaded is not None and uploaded.size > MAX_MB * 1024 * 1024:
    st.error(f"File too large (> {MAX_MB} MB).")
    st.stop()


run_btn = st.button("Run")


# -----------------------------
# Run pipeline
# -----------------------------
if run_btn:
    try:
        with st.spinner("Running…"):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                out_dir = tmp / "out"

                uploaded_bytes = uploaded.read() if uploaded else None
                uploaded_name = uploaded.name if uploaded else None

                out = run_job(
                    input_type=input_type,
                    smiles_text=smiles_text,
                    uploaded_bytes=uploaded_bytes,
                    uploaded_name=uploaded_name,
                    target_pH=target_pH,
                    output_name=output_name.strip() or "ligand",
                    output_format=output_format,
                    out_dir=str(out_dir),
                )

                st.success("Done!")

                # Summary
                st.text_area("Summary", out["summary_text"], height=220)

                # If multiple ligands (e.g., R/S or .smi list), choose one to visualize
                results = out["results"]
                names = [r["name"] for r in results]
                selected = st.selectbox("Select ligand to visualize", names, index=0)
                r = next(x for x in results if x["name"] == selected)

                if "warning" in r:
                    st.warning(r["warning"])

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("2D Structures")
                    show_smiles_2d(r["base_smiles"], "Base SMILES")
                    show_smiles_2d(r["ph_smiles"], f"pH-adjusted SMILES (pH={target_pH})")
                    st.write(f"Formal charge: **{r['formal_charge']}**")
                    if r["pka_pred"] is not None:
                        st.write(f"Predicted pKa (ML): **{r['pka_pred']:.2f}**")

                with col2:
                    st.subheader("3D Viewer (minimized)")
                    show_pdb_3d(r["pdb_text_for_viewer"])

                # Download buttons
                st.markdown("---")
                zip1 = tmp / "pkanet_outputs_all.zip"
                zip_all_outputs(str(out_dir), str(zip1))

                st.download_button(
                    "Download ALL outputs (ZIP)",
                    data=zip1.read_bytes(),
                    file_name="pkanet_outputs_all.zip",
                    mime="application/zip",
                )

                zip2 = tmp / "pkanet_outputs_minimized_only.zip"
                zip_minimized_only(str(out_dir), str(zip2))

                st.download_button(
                    "Download minimized outputs only (ZIP)",
                    data=zip2.read_bytes(),
                    file_name="pkanet_outputs_minimized_only.zip",
                    mime="application/zip",
                )

                # Show which file was written in selected format
                st.caption(f"Primary output file for **{selected}**: `{Path(r['output_file']).name}`")

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")


with st.expander("📌 Please cite us / Acknowledgements", expanded=False):
    st.markdown(
        """
### Please cite us

If you use **pKaNET Cloud** in your research, publications, or presentations,
please cite the following work:

**DFDD project**  
Hengphasatporn K *et al.*  
*Journal of Chemical Information and Modeling* (2026)

Proper citation helps us demonstrate impact and supports continued development
and maintenance of this tool.

---

### Acknowledgements (Special thanks)

This tool makes use of several excellent open-source projects and research efforts:

- **pKaPredict / pKaNET** — machine-learning–based pKa prediction framework  
- **Machine Learning Meets pKa (czodrowskilab)** — research and methodological
  inspiration for ML-based pKa estimation  
- **Dimorphite-DL** — pH-dependent protonation state enumeration  
  (Ropp PJ *et al.*, *J. Cheminformatics*, 2019)  
- **RDKit** — open-source cheminformatics toolkit for molecular handling,
  2D/3D structure generation, and energy minimization

We sincerely thank the authors and maintainers of these projects for providing
robust and well-documented foundations for **pKaNET Cloud**.
        """
    )


