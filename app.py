# app.py
import tempfile
from pathlib import Path

import streamlit as st
from rdkit import Chem

# RDKit Draw can fail on some Streamlit Cloud builds (rdMolDraw2D missing).
# Fall back gracefully: show SMILES text instead of 2D image.
try:
    from rdkit.Chem import Draw
    _HAS_DRAW = True
except Exception:
    Draw = None
    _HAS_DRAW = False

import py3Dmol
from stmol import showmol

from core import run_job, zip_all_outputs, zip_minimized_only


# =========================
# Page config + style
# =========================
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
    unsafe_allow_html=True,
)

st.title("pKaNET Cloud — pH-adjusted 3D builder + pKa prediction")

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
    # This requires Pka_WF.png to exist in your repo root
    try:
        st.image("Pka_WF.png", use_column_width=True)
    except Exception:
        st.info("Workflow image not found (Pka_WF.png).")


# =========================
# UI helpers
# =========================
def show_smiles_2d(smiles: str, title: str):
    st.markdown(f"**{title}**")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.warning("Cannot parse SMILES for 2D drawing.")
        st.code(smiles)
        return

    if not _HAS_DRAW:
        st.info("2D depiction is unavailable on this server build. Showing SMILES instead.")
        st.code(smiles)
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


# =========================
# Sidebar controls
# =========================
st.sidebar.header("Input")
input_type = st.sidebar.radio(
    "Choose input type",
    ["SMILES", "SMI_FILE", "FILE"],
    help="SMILES: single molecule. SMI_FILE: list of SMILES. FILE: upload PDB/MOL2/SDF.",
)

st.sidebar.header("Chemistry settings")
target_pH = st.sidebar.slider("Target pH", 2.0, 12.0, 7.0, 0.1)

output_format = st.sidebar.selectbox(
    "Output format",
    ["PDB", "SDF", "MOL2"],
    index=0,
    help="MOL2 may fall back to SDF depending on server support.",
)

st.sidebar.header("Naming")
output_name = st.sidebar.text_input(
    "Base output name",
    value="ligand",
    help="Used for single SMILES/FILE. For SMI_FILE, names come from the file.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="small-note">'
    'Outputs include: minimized structure, a viewer PDB, and a summary file.'
    '</div>',
    unsafe_allow_html=True,
)

with st.sidebar.expander("How to use (examples)", expanded=False):
    st.markdown(
        """
**1) Single SMILES**
- Select `SMILES`, paste a SMILES, choose pH and output format, then click **Run**.

**2) SMILES list (.smi)**
- Select `SMI_FILE` and upload a text file with one molecule per line:
```text
CCO ethanol
CC(=O)O acetic_acid
c1ccccc1 benzene

**3) Ligand file

- Select `FILE` and upload .pdb, .sdf, or .mol2.

"""
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: MOL2 may fall back to SDF depending on server build.")


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

            results = out["results"]
            names = [r["name"] for r in results]

            st.success("Done!")

            tab_overview, tab_view, tab_download = st.tabs(
                ["Overview", "2D / 3D View", "Downloads"]
            )

            with tab_overview:
                st.text_area("Summary", out["summary_text"], height=260)

            with tab_view:
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
                    if r.get("pka_pred") is not None:
                        st.write(f"Predicted pKa (ML): **{r['pka_pred']:.2f}**")

                with col2:
                    st.subheader("3D Viewer (minimized)")
                    show_pdb_3d(r["pdb_text_for_viewer"])

                st.caption(
                    f"Primary output file for **{selected}**: `{Path(r['output_file']).name}`"
                )

            with tab_download:
                zip_all = tmp / "pkanet_outputs_all.zip"
                zip_all_outputs(str(out_dir), str(zip_all))

                st.download_button(
                    "Download ALL outputs (ZIP)",
                    data=zip_all.read_bytes(),
                    file_name="pkanet_outputs_all.zip",
                    mime="application/zip",
                )

                zip_min = tmp / "pkanet_outputs_minimized_only.zip"
                zip_minimized_only(str(out_dir), str(zip_min))

                st.download_button(
                    "Download minimized outputs only (ZIP)",
                    data=zip_min.read_bytes(),
                    file_name="pkanet_outputs_minimized_only.zip",
                    mime="application/zip",
                )

except Exception as e:
    st.error(f"Error: {e}")


