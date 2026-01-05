# app.py
import streamlit as st
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw

import py3Dmol
from stmol import showmol

from core import run_job, zip_all_outputs, zip_minimized_only


st.set_page_config(page_title="pKaNET Cloud", layout="wide")
st.title("pKaNET Cloud — pH-adjusted 3D builder + pKa prediction")


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
st.sidebar.header("Input / Options")

input_type = st.sidebar.selectbox("Input type", ["SMILES", "SMI_FILE", "FILE"], index=0)
target_pH = st.sidebar.slider("Target pH", 2.0, 12.0, 7.0, 0.1)

output_format = st.sidebar.selectbox("Output format", ["PDB", "SDF", "MOL2"], index=0)

output_name = st.sidebar.text_input(
    "Output name (for single SMILES / FILE)",
    value="ligand"
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

