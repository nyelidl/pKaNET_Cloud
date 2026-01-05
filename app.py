import streamlit as st
import tempfile
from pathlib import Path
from core import run_job, zip_all_outputs, zip_minimized_pdb_only

st.set_page_config(page_title="pKaNET Cloud", layout="wide")
st.title("pKaNET Cloud — Protonation & 3D builder")

st.sidebar.header("Input / Options")
input_type = st.sidebar.selectbox("Input type", ["SMILES", "SMI_FILE", "FILE"])
target_pH = st.sidebar.slider("Target pH", 2.0, 12.0, 7.0, 0.1)
output_name = st.sidebar.text_input("Output name (for single SMILES/FILE)", value="ligand")

smiles_text = None
uploaded = None

if input_type == "SMILES":
    smiles_text = st.text_area("SMILES", height=120, placeholder="Paste a SMILES here")
elif input_type == "SMI_FILE":
    uploaded = st.file_uploader("Upload .smi (SMILES [name] per line)", type=["smi", "txt"])
else:
    uploaded = st.file_uploader("Upload ligand file", type=["pdb", "mol2", "sdf"])

run_btn = st.button("Run")

if run_btn:
    try:
        with st.spinner("Running..."):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)

                uploaded_bytes = uploaded.read() if uploaded else None
                uploaded_name = uploaded.name if uploaded else None

                out_dir = tmp / "out"
                out = run_job(
                    input_type=input_type,
                    smiles_text=smiles_text,
                    uploaded_bytes=uploaded_bytes,
                    uploaded_name=uploaded_name,
                    target_pH=target_pH,
                    output_name=output_name,
                    out_dir=str(out_dir),
                )

                st.success("Done!")
                st.text_area("Summary", out["summary_text"], height=220)

                # ZIP everything
                zip_all = tmp / "all_outputs.zip"
                zip_all_outputs(str(out_dir), str(zip_all))
                st.download_button(
                    "Download ALL outputs (ZIP)",
                    data=zip_all.read_bytes(),
                    file_name="pkanet_outputs.zip",
                    mime="application/zip",
                )

                # ZIP only minimized PDB
                zip_pdb = tmp / "minimized_pdb_only.zip"
                zip_minimized_pdb_only(str(out_dir), str(zip_pdb))
                st.download_button(
                    "Download minimized PDB only (ZIP)",
                    data=zip_pdb.read_bytes(),
                    file_name="minimized_pdb_outputs.zip",
                    mime="application/zip",
                )

    except Exception as e:
        st.error(f"Error: {e}")

