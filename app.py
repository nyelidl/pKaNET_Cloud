import streamlit as st
import tempfile
import subprocess
import shutil
from pathlib import Path
from core import (
    run_job, zip_all_outputs, zip_minimized_structures, DISPLAY_COLS, _PKA_BACKEND
)
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from rdkit.Chem import Draw
    DRAW_AVAILABLE = True
except (ImportError, OSError):
    DRAW_AVAILABLE = False
    class Draw:
        @staticmethod
        def MolToImage(*a, **kw):
            return None

st.set_page_config(page_title="pKaNET Cloud+", layout="wide", page_icon="🧪")

st.markdown("""
<style>
.main-header { font-size:2.5rem; font-weight:bold; color:#1f77b4;
               text-align:center; margin-bottom:0.5rem; }
.sub-header  { text-align:center; color:#666; margin-bottom:2rem; }
.stDownloadButton button { width:100%; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧪 pKaNET Cloud+</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Tautomer-aware microstate ranking · Henderson–Hasselbalch scoring · '
    'PubChem pKa evidence · pH-adjusted 3D structures<br>'
    'Part of the <a href="https://github.com/nyelidl/DFDD" target="_blank">'
    '<strong>DFDD Project</strong></a>'
    '</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# PDB → SMILES conversion
# ─────────────────────────────────────────────────────────────────────────────

def pdb_to_canonical_smiles(pdb_bytes: bytes):
    if not shutil.which("obabel"):
        return None, "Open Babel (obabel) not found. Install: conda install -c conda-forge openbabel"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pdb_file = tmp / "input.pdb"
            smi_file = tmp / "output.smi"
            pdb_file.write_bytes(pdb_bytes)
            r = subprocess.run(
                ["obabel", str(pdb_file), "-O", str(smi_file), "--canonical"],
                capture_output=True, text=True, timeout=30,
            )
            if not smi_file.exists() or smi_file.stat().st_size == 0:
                return None, f"obabel produced no output. stderr: {r.stderr.strip() or '(none)'}"
            raw    = smi_file.read_text(encoding="utf-8", errors="replace").strip()
            smiles = raw.splitlines()[0].split()[0].strip() if raw else ""
            return (smiles or None), (None if smiles else "Could not parse SMILES from obabel output.")
    except subprocess.TimeoutExpired:
        return None, "obabel timed out (>30 s)."
    except Exception as e:
        return None, f"Error: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Input / Options")
input_type  = st.sidebar.selectbox("Input type", ["SMILES", "SMI_FILE", "FILE"])
target_pH   = st.sidebar.slider("Target pH", 2.0, 12.0, 7.4, 0.1)
output_name = st.sidebar.text_input("Output name", value="ligand")

st.sidebar.header("🧬 Stereochemistry")
stereo_mode = st.sidebar.selectbox(
    "Stereochemistry", ["Enumerate R/S", "Keep as-is"],
    help="Enumerate R/S: generate both stereoisomers for undefined chiral centers.",
)
keep_stereo = (stereo_mode == "Keep as-is")

st.sidebar.header("🔀 Microstate Settings")
ph_window = st.sidebar.slider(
    "pH window", 0.2, 2.0, 1.0, 0.1,
    help="Dimorphite-DL enumerates states in [pH − window/2, pH + window/2].",
)
max_tautomers = st.sidebar.slider("Max tautomers", 1, 20, 8)
top_n_microstates = st.sidebar.slider("Top N microstates", 1, 10, 5)
write_alt_3d_for_top_k = st.sidebar.slider("Write 3D for top-k", 1, 5, 3)

st.sidebar.header("🔬 PubChem pKa Evidence")
use_pubchem = st.sidebar.checkbox(
    "Query PubChem for experimental pKa", value=True,
    help="Fetch dissociation constants from PubChem to guide microstate scoring.",
)

st.sidebar.header("📄 Output Format")
output_formats = st.sidebar.multiselect("Formats", ["PDB", "MOL2"], default=["PDB"],
                                         help="SDF is always generated for 3D visualization.")
if not output_formats:
    st.sidebar.warning("⚠️ Please select at least one output format")

st.sidebar.header("🎨 Visualization")
show_2d = st.sidebar.checkbox("Show 2D structure", value=True) if DRAW_AVAILABLE else False
if not DRAW_AVAILABLE:
    st.sidebar.info("ℹ️ 2D visualization not available on this server")
show_3d       = st.sidebar.checkbox("Show 3D structure", value=True)
viewer_width  = st.sidebar.slider("3D Viewer Width",  300, 800, 460, 20)
viewer_height = st.sidebar.slider("3D Viewer Height", 200, 600, 360, 20)

# ─────────────────────────────────────────────────────────────────────────────
# Input widgets
# ─────────────────────────────────────────────────────────────────────────────

smiles_text = None
uploaded    = None

if input_type == "SMILES":
    smiles_text = st.text_area(
        "SMILES   example: CC(=O)OC1=CC=CC=C1C(=O)O",
        value="CC(=O)OC1=CC=CC=C1C(=O)O",
        height=100,
    )
elif input_type == "SMI_FILE":
    uploaded = st.file_uploader("Upload .smi file (SMILES [name] per line)", type=["smi", "txt"])
    st.info("📝 Format: `SMILES [optional_name]` per line")
else:
    uploaded = st.file_uploader("Upload ligand file", type=["pdb", "mol2", "sdf"])
    st.info("📝 PDB files are converted to canonical SMILES via Open Babel before processing.")

# Live PDB preview
converted_smiles_from_pdb = None
if input_type == "FILE" and uploaded is not None and uploaded.name.endswith(".pdb"):
    pdb_bytes = uploaded.read()
    converted_smiles_from_pdb, conv_err = pdb_to_canonical_smiles(pdb_bytes)
    if conv_err:
        st.error(f"❌ PDB → SMILES failed: {conv_err}")
    else:
        st.success("✅ PDB converted to canonical SMILES")
        st.code(converted_smiles_from_pdb, language="text")
        if DRAW_AVAILABLE:
            mol_p = Chem.MolFromSmiles(converted_smiles_from_pdb)
            if mol_p:
                AllChem.Compute2DCoords(mol_p)
                img_p = Draw.MolToImage(mol_p, size=(400, 280))
                if img_p:
                    st.image(img_p, caption="2D preview")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: 3D viewer
# ─────────────────────────────────────────────────────────────────────────────

def create_3dmol_viewer(sdf_content: str, width: int, height: int) -> str:
    return f"""
    <div id="container" style="width:{width}px;height:{height}px;position:relative;"></div>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <script>
        let viewer = $3Dmol.createViewer(
            document.getElementById('container'), {{backgroundColor:'white'}});
        viewer.addModel(`{sdf_content}`, "sdf");
        viewer.setStyle({{}}, {{stick:{{radius:0.2}}}});
        viewer.zoomTo();
        viewer.render();
    </script>"""

# ─────────────────────────────────────────────────────────────────────────────
# Helper: microstate table
# ─────────────────────────────────────────────────────────────────────────────

def show_microstate_table(top_microstates: list) -> None:
    try:
        import pandas as pd
    except ImportError:
        st.warning("pandas not available.")
        return
    cols = [c for c in DISPLAY_COLS if c in top_microstates[0]]
    df   = pd.DataFrame([{c: r.get(c, "") for c in cols} for r in top_microstates])
    st.dataframe(df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: display one ligand result
# ─────────────────────────────────────────────────────────────────────────────

def display_ligand_result(r: dict, idx: int = 0) -> None:
    t = (r["top_microstates"] or [{}])[0]

    st.subheader("🏆 Rank-1 Microstate")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Score:** `{r['selection_score']:.3f}`")
        st.markdown(f"**SMILES:** `{r['selected_microstate_smiles']}`")
        st.markdown(f"**Charge @ pH {target_pH}:** `{r['formal_charge']:+d}`")
        st.markdown(f"**Charged atoms:** `{r['charged_atoms']}`")
        st.markdown(f"**Zwitterion (strict):** `{'YES 🧷' if r['is_zwitterion'] else 'NO'}`")
        st.markdown(f"**Aromaticity:** `{'LOST ⚠️' if r.get('flag_aromaticity_lost') else 'OK ✅'}`")
    with col2:
        st.markdown(f"**pKa source:** `{r['pKa_source']}`")
        st.markdown(f"**Backend:** `{r['decision_backend']}` ({r['decision_mode']})")
        if r.get("pubchem_cid"):
            st.markdown(f"**PubChem CID:** `{r['pubchem_cid']}`  "
                        f"**pKa:** `{r['pubchem_pka_values']}`  "
                        f"(conf=`{r['pubchem_confidence']}`)")
        else:
            st.markdown("**PubChem pKa:** `not found`")
        st.markdown(f"**Amide kept:** `{'YES ✅' if r['flag_amide_preserved'] else 'NO'}`")
        st.markdown(f"**Imidic acid:** `{'YES ⚠️' if r['flag_imidic_acid_penalty'] else 'NO'}`")
        st.markdown(f"**[N⁻]C=O:** `{'YES ⚠️' if r['flag_amide_n_deprotonation'] else 'NO'}`")

    # Flags row
    flag_col1, flag_col2, flag_col3 = st.columns(3)
    with flag_col1:
        if r["ambiguous_top_assignment"]:
            st.warning("⚠️ Ambiguous top state")
        else:
            st.success("✅ Unambiguous")
    with flag_col2:
        if r["flag_tautomer_rich"]:
            motifs = ", ".join(r.get("flag_tautomer_motifs", []))
            st.warning(f"🔄 Tautomer-rich: {motifs}")
        else:
            st.success("✅ No tautomer-rich motifs")
    with flag_col3:
        if r.get("flag_borderline_pka"):
            st.warning("⚠️ Borderline pKa (|pH–pKa| ≤ 1)")
        else:
            st.success("✅ pH well away from pKa")
        st.caption(f"{r.get('n_all_microstates', 0)} microstates evaluated")

    # Microstate table
    top_microstates = r.get("top_microstates", [])
    if top_microstates:
        with st.expander(f"📊 Ranked microstate table (top {len(top_microstates)})", expanded=True):
            show_microstate_table(top_microstates)
        if r.get("microstate_csv") and Path(r["microstate_csv"]).exists():
            st.download_button(
                "⬇️ Download microstate CSV",
                data=Path(r["microstate_csv"]).read_bytes(),
                file_name=f"{r['name']}_microstates.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"dl_csv_{idx}",
            )

    # 2D / 3D visualization
    if show_2d or show_3d:
        st.subheader("🎨 Structure Visualization")
        alt3d = r.get("alt3d", [])

        if show_3d and len(alt3d) > 1:
            tabs = st.tabs([f"Rank {d['rank']}" for d in alt3d])
            for tab, d in zip(tabs, alt3d):
                with tab:
                    viz_l, viz_r = st.columns(2) if show_2d else (None, st)
                    if show_2d and viz_l:
                        with viz_l:
                            st.markdown("**2D**")
                            mol2d = Chem.MolFromSmiles(d["smiles"])
                            if mol2d and DRAW_AVAILABLE:
                                AllChem.Compute2DCoords(mol2d)
                                img = Draw.MolToImage(mol2d, size=(400, 300))
                                if img:
                                    st.image(img, use_container_width=True)
                    with (viz_r if show_2d else tab):
                        st.markdown("**3D**")
                        sdf_p = d.get("sdf")
                        if sdf_p and Path(sdf_p).exists():
                            try:
                                html = create_3dmol_viewer(Path(sdf_p).read_text(),
                                                           viewer_width, viewer_height)
                                components.html(html, height=viewer_height + 20, scrolling=False)
                            except Exception as e:
                                st.warning(f"3D view failed: {e}")
                        st.caption(f"`{d['smiles']}`")
        else:
            viz_l, viz_r = st.columns(2) if (show_2d and show_3d) else (None, None)
            sdf_p = r.get("minimized_sdf")

            if show_2d:
                target_col = viz_l if viz_r else st
                with target_col:
                    st.markdown("**2D Structure (rank-1)**")
                    mol2d = Chem.MolFromSmiles(r["selected_microstate_smiles"])
                    if mol2d and DRAW_AVAILABLE:
                        AllChem.Compute2DCoords(mol2d)
                        img = Draw.MolToImage(mol2d, size=(400, 300))
                        if img:
                            st.image(img, use_container_width=True)

            if show_3d:
                target_col = viz_r if viz_l else st
                with target_col:
                    st.markdown("**3D Structure (rank-1)**")
                    if sdf_p and Path(sdf_p).exists():
                        try:
                            html = create_3dmol_viewer(Path(sdf_p).read_text(),
                                                       viewer_width, viewer_height)
                            components.html(html, height=viewer_height + 20, scrolling=False)
                        except Exception as e:
                            st.warning(f"3D view failed: {e}")
                    else:
                        st.warning("SDF file not found")

    # Output files
    with st.expander("📁 Output files"):
        for label, key in [("PDB (rank-1)", "minimized_pdb"),
                            ("MOL2 (rank-1)", "minimized_mol2"),
                            ("SDF (rank-1)", "minimized_sdf")]:
            fp = r.get(key)
            if fp and Path(fp).exists():
                st.markdown(f"- **{label}:** `{Path(fp).name}`")

# ─────────────────────────────────────────────────────────────────────────────
# Run button
# ─────────────────────────────────────────────────────────────────────────────

run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

if run_btn:
    if   input_type == "SMILES"   and not smiles_text:
        st.error("⚠️ Please enter a SMILES string")
    elif input_type == "SMI_FILE" and not uploaded:
        st.error("⚠️ Please upload a .smi file")
    elif input_type == "FILE"     and not uploaded:
        st.error("⚠️ Please upload a ligand file")
    elif input_type == "FILE" and uploaded.name.endswith(".pdb") and converted_smiles_from_pdb is None:
        st.error("⚠️ PDB → SMILES conversion failed. Cannot proceed.")
    elif not output_formats:
        st.error("⚠️ Please select at least one output format")
    else:
        try:
            with st.spinner("🔬 Enumerating tautomers, scoring microstates, building 3D structures…"):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)

                    if input_type == "FILE" and converted_smiles_from_pdb:
                        eff_type   = "SMILES"
                        eff_smiles = converted_smiles_from_pdb
                        eff_bytes  = None
                        eff_name   = None
                        st.info(f"🔄 Using SMILES from PDB: `{converted_smiles_from_pdb}`")
                    elif input_type == "FILE":
                        eff_type   = "FILE"
                        eff_smiles = None
                        eff_bytes  = uploaded.read()
                        eff_name   = uploaded.name
                    elif input_type == "SMILES":
                        eff_type   = "SMILES"
                        eff_smiles = smiles_text
                        eff_bytes  = None
                        eff_name   = None
                    else:
                        eff_type   = "SMI_FILE"
                        eff_smiles = None
                        eff_bytes  = uploaded.read()
                        eff_name   = uploaded.name

                    out = run_job(
                        input_type              = eff_type,
                        smiles_text             = eff_smiles,
                        uploaded_bytes          = eff_bytes,
                        uploaded_name           = eff_name,
                        target_pH               = target_pH,
                        output_name             = output_name,
                        out_dir                 = str(tmp / "out"),
                        output_formats          = output_formats,
                        enumerate_stereoisomers = not keep_stereo,
                        use_pubchem             = use_pubchem,
                        ph_window               = ph_window,
                        max_tautomers           = max_tautomers,
                        top_n_microstates       = top_n_microstates,
                        write_alt_3d_for_top_k  = write_alt_3d_for_top_k,
                    )

                    # FIX: was checking != "fine" which never matched; "none" is the
                    # correct sentinel when no ML backend is available.
                    backend_display = (out['pka_backend']
                                       if out['pka_backend'] != "none"
                                       else "heuristic (SMARTS table)")
                    st.success(f"✅ Analysis complete!  |  pKa backend: `{backend_display}`")

                    if out.get("format_warnings"):
                        with st.expander("⚠️ Warnings", expanded=False):
                            for w in out["format_warnings"]:
                                if w.startswith("ℹ️"):
                                    st.info(w)
                                else:
                                    st.warning(w)

                    with st.expander("📊 Summary", expanded=True):
                        st.text(out["summary_text"])

                    st.header("📈 Results")
                    results = out["results"]
                    if len(results) > 1:
                        tabs = st.tabs([r["name"] for r in results])
                        for i, (tab, r) in enumerate(zip(tabs, results)):
                            with tab:
                                display_ligand_result(r, idx=i)
                    else:
                        display_ligand_result(results[0], idx=0)

                    st.header("💾 Downloads")
                    out_dir_path = Path(out["out_dir"])

                    log_file = out_dir_path / "processing.log"
                    if log_file.exists():
                        st.download_button(
                            "📄 Download Processing Log",
                            data=log_file.read_bytes(),
                            file_name="pkanet_processing.log",
                            mime="text/plain",
                            use_container_width=True,
                        )
                        st.markdown("---")

                    col1, col2 = st.columns(2)
                    with col1:
                        zip_all = tmp / "all_outputs.zip"
                        zip_all_outputs(str(out_dir_path), str(zip_all))
                        st.download_button(
                            "📦 Download ALL outputs (ZIP)",
                            data=zip_all.read_bytes(),
                            file_name="pkanet_all_outputs.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )
                    with col2:
                        zip_sel = tmp / "structures.zip"
                        zip_minimized_structures(str(out_dir_path), str(zip_sel), output_formats)
                        label = (f"🧬 Download {output_formats[0]} files (ZIP)"
                                 if len(output_formats) == 1
                                 else f"🧬 Download {' + '.join(output_formats)} (ZIP)")
                        st.download_button(
                            label,
                            data=zip_sel.read_bytes(),
                            file_name=f"pkanet_{'_'.join(f.lower() for f in output_formats)}.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — About / Citation / Examples
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown(f"### ℹ️ pKa backend: `{_PKA_BACKEND}`")
st.sidebar.info("""
**Pipeline:**
1. SMILES standardization (RDKit)
2. PubChem experimental pKa lookup
3. Tautomer enumeration + SMARTS plausibility scoring
4. Dimorphite-DL protonation state enumeration
5. Henderson–Hasselbalch microstate scoring (5 layers)
6. Rank & select top microstate
7. RDKit ETKDGv3 + MMFF/UFF 3D minimization

**pKa backends (priority order):**
pkasolver → propka → unipka CLI → heuristic SMARTS table
""")

st.sidebar.markdown("### 📚 Citation")
st.sidebar.markdown("""
- DFDD: Hengphasatporn K. et al. JCIM (2026)
- Anyone Can Dock: Hengphasatporn K. et al. JCIM (2026)
- Dimorphite-DL: Ropp PJ et al. J Cheminform (2019)
""")

st.sidebar.markdown("### 💡 Examples")
st.sidebar.markdown("""
**Acylhydrazone** (amide NH preserved, charge 0):
```
O=C(N/N=C/CC)C1=NC(C(N/N=C/CC)=O)=CC=C1
```
**Glycine** (zwitterion at pH 7.4):
```
NCC(=O)O
```
**Apigenin** (7-OH deprotonated, charge −1):
```
O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12
```
**Baicalein** (7-OH deprotonated, charge −1):
```
O=c1cc(-c2ccccc2)oc2cc(O)c(O)c(O)c12
```
**Osimertinib** (dimethylaminoalkyl, charge +1):
```
COc1cc2c(cc1NC(=O)/C=C/CN(C)C)ncnc2Nc1ccc(F)c(Cl)c1
```
**Kaempferol** (3-OH flavonol, 7-OH deprotonated, charge −1):
```
O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12
```
""")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666;font-size:0.9rem;'>"
    "🧬 pKaNET Cloud+ — "
    "<a href='mailto:kowith@ccs.tsukuba.ac.jp'>kowith@ccs.tsukuba.ac.jp</a>"
    "</div>",
    unsafe_allow_html=True,
)
