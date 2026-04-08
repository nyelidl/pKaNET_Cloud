import streamlit as st
import tempfile
import subprocess
import shutil
from pathlib import Path
from core import run_job, zip_all_outputs, zip_minimized_structures
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem
import io

# Fix for RDKit Draw on headless servers
try:
    from rdkit.Chem import Draw
    DRAW_AVAILABLE = True
except (ImportError, OSError) as e:
    DRAW_AVAILABLE = False
    print(f"Warning: RDKit Draw not available: {e}")
    class DrawFallback:
        @staticmethod
        def MolToImage(*args, **kwargs):
            return None
    Draw = DrawFallback()

st.set_page_config(page_title="pKaNET Cloud", layout="wide", page_icon="🧪")

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4;
                   text-align: center; margin-bottom: 0.5rem; }
    .sub-header  { text-align: center; color: #666; margin-bottom: 2rem; }
    .result-card { background-color: #f0f2f6; padding: 1rem;
                   border-radius: 0.5rem; margin: 1rem 0; }
    .stDownloadButton button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧪 pKaNET Cloud</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Machine-Learning–Driven Protonation & pH-Aware 3D Structure Generation<br>'
    '<span style="font-size:0.9em;">'
    'Tautomer-aware microstate ranking · PubChem pKa evidence · '
    'HH scoring · R/S stereoisomer enumeration'
    '</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">'
    'Part of the <a href="https://github.com/nyelidl/DFDD" target="_blank">'
    '<strong>DFDD Project</strong></a>.'
    '</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# PDB → canonical SMILES conversion
# ─────────────────────────────────────────────────────────────────────────────

def pdb_to_canonical_smiles(pdb_bytes: bytes):
    if shutil.which("obabel") is None:
        return None, (
            "Open Babel (obabel) is not installed or not found in PATH. "
            "Please install it: conda install -c conda-forge openbabel"
        )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pdb_file = tmp / "input.pdb"
            smi_file = tmp / "output.smi"
            pdb_file.write_bytes(pdb_bytes)
            result = subprocess.run(
                ["obabel", str(pdb_file), "-O", str(smi_file), "--canonical"],
                capture_output=True, text=True, timeout=30,
            )
            if not smi_file.exists() or smi_file.stat().st_size == 0:
                return None, f"obabel produced no output. stderr: {result.stderr.strip() or '(none)'}"
            raw = smi_file.read_text(encoding="utf-8", errors="replace").strip()
            if not raw:
                return None, "obabel produced an empty SMILES file."
            smiles = raw.splitlines()[0].split()[0].strip()
            return (smiles or None), (None if smiles else "Could not parse SMILES from obabel output.")
    except subprocess.TimeoutExpired:
        return None, "obabel conversion timed out (>30 s)."
    except Exception as exc:
        return None, f"Unexpected error: {exc}"

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Input / Options")
input_type  = st.sidebar.selectbox("Input type", ["SMILES", "SMI_FILE", "FILE"])
target_pH   = st.sidebar.slider("Target pH", 2.0, 12.0, 7.4, 0.1)
output_name = st.sidebar.text_input("Output name (single SMILES/FILE)", value="ligand")

st.sidebar.header("🧬 Stereochemistry")
enumerate_stereoisomers = st.sidebar.checkbox(
    "Enumerate R/S stereoisomers", value=True,
    help="Automatically generate both R and S stereoisomers for undefined chiral centers.",
)

st.sidebar.header("🔀 Microstate Settings")
ph_window = st.sidebar.slider(
    "pH window (±½ around target pH)", 0.2, 2.0, 1.0, 0.1,
    help="Dimorphite-DL enumerates protonation states in [pH − window/2, pH + window/2].",
)
max_tautomers = st.sidebar.slider(
    "Max tautomers to enumerate", 1, 20, 8,
    help="Maximum tautomers to score and carry forward.",
)
top_n_microstates = st.sidebar.slider(
    "Top N microstates to report", 1, 10, 5,
    help="Number of ranked candidate microstates displayed in the results table.",
)
write_alt_3d_for_top_k = st.sidebar.slider(
    "Write 3D for top-k microstates", 1, 5, 3,
    help="Generate minimized 3D files for the top-k ranked microstates.",
)

st.sidebar.header("🔬 pKa & Evidence")
use_iupac_pka = st.sidebar.checkbox(
    "Use IUPAC pKa database (when available)", value=True,
    help="Try IUPAC high-confidence pKa dataset first, then pKaPredict ML model.",
)
use_pubchem = st.sidebar.checkbox(
    "Use PubChem experimental pKa evidence", value=True,
    help="Query PubChem dissociation constant data to guide microstate scoring.",
)

st.sidebar.header("📄 Output Format")
output_formats = st.sidebar.multiselect(
    "Select output formats", ["PDB", "MOL2"], default=["PDB"],
    help="SDF is always generated for 3D visualization.",
)
if not output_formats:
    st.sidebar.warning("⚠️ Please select at least one output format")

st.sidebar.header("🎨 Visualization Options")
if DRAW_AVAILABLE:
    show_2d = st.sidebar.checkbox("Show 2D structure", value=True)
else:
    show_2d = False
    st.sidebar.info("ℹ️ 2D visualization not available on this server")
show_3d = st.sidebar.checkbox("Show 3D structure", value=True)
viewer_width  = st.sidebar.slider("3D Viewer Width",  300, 800, 300, 50)
viewer_height = st.sidebar.slider("3D Viewer Height", 200, 600, 300, 50)

# ─────────────────────────────────────────────────────────────────────────────
# Input widgets
# ─────────────────────────────────────────────────────────────────────────────

smiles_text = None
uploaded    = None

if input_type == "SMILES":
    smiles_text = st.text_area(
        "SMILES\nexample: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        height=120,
        placeholder="Paste a SMILES string here:",
    )
elif input_type == "SMI_FILE":
    uploaded = st.file_uploader("Upload .smi (SMILES [name] per line)", type=["smi", "txt"])
    st.info("📝 Format: `SMILES [optional_name]` per line")
else:  # FILE (PDB)
    uploaded = st.file_uploader("Upload ligand PDB file", type=["pdb"])
    st.info("📝 The PDB file will be automatically converted to canonical SMILES using Open Babel.")

# Live PDB → SMILES preview
converted_smiles_from_pdb = None
if input_type == "FILE" and uploaded is not None:
    pdb_bytes = uploaded.read()
    converted_smiles_from_pdb, conv_err = pdb_to_canonical_smiles(pdb_bytes)
    if conv_err:
        st.error(f"❌ PDB → SMILES conversion failed: {conv_err}")
    else:
        st.success("✅ PDB converted to canonical SMILES successfully")
        st.markdown("**Canonical SMILES extracted from PDB:**")
        st.code(converted_smiles_from_pdb, language="text")
        if DRAW_AVAILABLE:
            mol_preview = Chem.MolFromSmiles(converted_smiles_from_pdb)
            if mol_preview:
                AllChem.Compute2DCoords(mol_preview)
                img_preview = Draw.MolToImage(mol_preview, size=(400, 280))
                if img_preview:
                    st.image(img_preview, caption="2D preview of extracted structure")

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def draw_molecule_2d(smiles_str, size=(400, 300)):
    if not DRAW_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        return Draw.MolToImage(mol, size=size)
    except Exception as e:
        st.warning(f"Could not generate 2D structure: {e}")
        return None


def create_3dmol_viewer(sdf_content, width=400, height=300):
    return f"""
    <div id="container" style="width: {width}px; height: {height}px; position: relative;"></div>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById('container'), {{
            backgroundColor: 'white'
        }});
        let sdfData = `{sdf_content}`;
        viewer.addModel(sdfData, "sdf");
        viewer.setStyle({{}}, {{stick: {{radius: 0.2}}}});
        viewer.zoomTo();
        viewer.render();
    </script>
    """


def display_microstate_table(top_microstates: list) -> None:
    """Render a compact summary of the ranked microstate table."""
    try:
        import pandas as pd
    except ImportError:
        st.warning("pandas not available for microstate table.")
        return

    COLS = [
        "microstate_rank", "microstate_smiles", "selection_score", "delta_from_best",
        "net_charge", "charged_atoms", "is_zwitterion_strict",
        "flag_amide_preserved", "flag_imidic_acid_penalty",
        "flag_amide_n_deprotonation_penalty", "flag_borderline_pka",
        "tautomer_plausibility", "decision_backend", "pKa_source",
    ]
    rows = [{k: r.get(k, "") for k in COLS} for r in top_microstates]
    df   = pd.DataFrame(rows)

    # Friendly column display names
    rename = {
        "microstate_rank":                   "Rank",
        "microstate_smiles":                 "SMILES",
        "selection_score":                   "Score",
        "delta_from_best":                   "ΔScore",
        "net_charge":                        "Charge",
        "charged_atoms":                     "Charged atoms",
        "is_zwitterion_strict":              "Zwitterion",
        "flag_amide_preserved":              "Amide ✓",
        "flag_imidic_acid_penalty":          "Imidic ⚠",
        "flag_amide_n_deprotonation_penalty":"[N⁻]C=O ⚠",
        "flag_borderline_pka":               "Borderline pKa",
        "tautomer_plausibility":             "Tautomer score",
        "decision_backend":                  "Backend",
        "pKa_source":                        "pKa source",
    }
    df = df.rename(columns=rename)
    st.dataframe(df, use_container_width=True, hide_index=True)


def display_ligand_result(result, out_dir, show_2d, show_3d, viewer_width, viewer_height) -> None:
    """Display full results for a single ligand."""

    # ── Molecular information ────────────────────────────────────────────────
    st.subheader("🔬 Molecular Information")
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown(f"**Name:** `{result['name']}`")
        st.markdown(f"**Base SMILES:** `{result['base_smiles']}`")
        st.markdown(f"**Rank-1 SMILES (pH {target_pH}):** `{result['ph_smiles']}`")
        st.markdown(f"**Charged atoms:** `{result.get('charged_atoms', 'none')}`")

    with info_col2:
        st.markdown(f"**Target pH:** `{target_pH}`")
        if result["pka_pred"] is not None:
            st.markdown(f"**Predicted pKa ({result.get('pka_source', '?')}):** `{result['pka_pred']:.2f}`")
        else:
            st.markdown("**Predicted pKa:** `N/A` ⚠️")

        if result.get("pubchem_pka"):
            pc_vals = result["pubchem_pka"]
            pc_conf = result.get("pubchem_confidence", "n/a")
            st.markdown(f"**PubChem pKa** (CID={result.get('pubchem_cid')}, conf={pc_conf}): "
                        f"`{', '.join(f'{v:.2f}' for v in pc_vals)}`")
        else:
            st.markdown("**PubChem pKa:** `not found`")

        st.markdown(f"**Net formal charge:** `{result['formal_charge']:+d}`")
        st.markdown(f"**Zwitterion (strict):** `{'YES 🧷' if result.get('is_zwitterion') else 'NO'}`")
        if result.get("stereoisomer_id"):
            st.markdown(f"**Stereoisomer:** `{result['stereoisomer_id']}`")

    # ── Flags & quality indicators ───────────────────────────────────────────
    st.subheader("🚦 Quality Flags")
    flag_col1, flag_col2, flag_col3, flag_col4 = st.columns(4)

    amb   = result.get("ambiguous", False)
    tr    = result.get("tautomer_rich", False)
    amide = result.get("flag_amide_preserved", False)
    imid  = result.get("flag_imidic_acid_penalty", False)
    amid2 = result.get("flag_amide_n_deprotonation", False)
    bord  = result.get("flag_borderline_pka", False)
    score = result.get("selection_score")
    be    = result.get("decision_backend", "heuristic")

    with flag_col1:
        if amb:
            st.warning("⚠️ **Ambiguous** top state\nMultiple microstates have similar scores.")
        else:
            st.success("✅ Unambiguous top state")
        if tr:
            motifs = ", ".join(result.get("tautomer_motifs", []))
            st.warning(f"🔄 **Tautomer-rich** ({motifs})")
        else:
            st.success("✅ No tautomer-rich motifs")

    with flag_col2:
        if amide:
            st.success("✅ Amide bond preserved")
        if imid:
            st.error("❌ Imidic acid penalty")
        if amid2:
            st.error("❌ Amide-N deprotonation [N⁻]C=O")
        if not imid and not amid2:
            st.success("✅ No chemistry flags")

    with flag_col3:
        if bord:
            st.warning("⚠️ Borderline pKa\n(|pH – pKa| ≤ 1)")
        else:
            st.success("✅ pH well away from pKa")
        n_all = result.get("n_all_microstates", 0)
        st.info(f"🔢 {n_all} microstates evaluated")

    with flag_col4:
        if score is not None:
            st.metric("Selection score (rank-1)", f"{score:.3f}")
        st.caption(f"pKa backend: `{be}`")

    # ── Ranked microstate table ───────────────────────────────────────────────
    top_microstates = result.get("top_microstates", [])
    if top_microstates:
        with st.expander(f"📊 Ranked Microstate Table (top {len(top_microstates)})", expanded=True):
            display_microstate_table(top_microstates)
            if result.get("microstate_csv") and Path(result["microstate_csv"]).exists():
                st.download_button(
                    "⬇️ Download full microstate CSV",
                    data=Path(result["microstate_csv"]).read_bytes(),
                    file_name=f"{result['name']}_microstates.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # ── 3D viewer for each alt 3D (if multiple) ──────────────────────────────
    alt_3d = result.get("alt_3d", [])
    if len(alt_3d) > 1 and show_3d:
        st.subheader("🔭 Top-k 3D Structures")
        tabs_3d = st.tabs([f"Rank {d['rank']}" for d in alt_3d])
        for tab, d in zip(tabs_3d, alt_3d):
            with tab:
                sdf_path = d["files"].get("sdf")
                if sdf_path and Path(sdf_path).exists():
                    try:
                        sdf_content = Path(sdf_path).read_text()
                        viewer_html = create_3dmol_viewer(sdf_content, width=viewer_width, height=viewer_height)
                        components.html(viewer_html, height=viewer_height + 20, scrolling=False)
                        st.caption(f"SMILES: `{d['smiles']}`")
                    except Exception as e:
                        st.warning(f"3D view failed: {e}")
                else:
                    st.warning("SDF not found for this microstate")

    elif (show_2d or show_3d):
        # Single-result visualization
        st.subheader("🎨 Structure Visualization")

        viz_col1, viz_col2 = st.columns(2) if (show_2d and show_3d) else (None, None)

        def _render_2d(col):
            with col:
                st.markdown("**2D Structure (rank-1)**")
                if DRAW_AVAILABLE:
                    img = draw_molecule_2d(result["ph_smiles"], size=(400, 300))
                    if img:
                        st.image(img, use_container_width=True)
                    else:
                        st.warning("Could not generate 2D structure")

        def _render_3d(col):
            with col:
                st.markdown("**3D Structure (rank-1)**")
                sdf_path = result.get("minimized_sdf")
                if sdf_path and Path(sdf_path).exists():
                    try:
                        sdf_content = Path(sdf_path).read_text()
                        viewer_html = create_3dmol_viewer(sdf_content, width=viewer_width, height=viewer_height)
                        components.html(viewer_html, height=viewer_height + 20, scrolling=False)
                    except Exception as e:
                        st.warning(f"3D visualization failed: {e}")
                else:
                    st.warning("3D structure file not found")

        if show_2d and show_3d:
            _render_2d(viz_col1)
            _render_3d(viz_col2)
        elif show_2d:
            _render_2d(st)
        elif show_3d:
            _render_3d(st)

    # ── Output files ────────────────────────────────────────────────────────
    with st.expander("📁 Output Files"):
        available_files = []
        if result.get("minimized_pdb") and Path(result["minimized_pdb"]).exists():
            available_files.append(f"- **PDB (rank-1):** `{Path(result['minimized_pdb']).name}`")
        if result.get("minimized_mol2") and Path(result["minimized_mol2"]).exists():
            available_files.append(f"- **MOL2 (rank-1):** `{Path(result['minimized_mol2']).name}`")
        if result.get("minimized_sdf") and Path(result["minimized_sdf"]).exists():
            available_files.append(f"- **SDF (rank-1):** `{Path(result['minimized_sdf']).name}` (for visualization)")
        for d in result.get("alt_3d", [])[1:]:
            for fmt, fp in d["files"].items():
                if fmt != "png_2d" and fp and Path(fp).exists():
                    available_files.append(f"- **{fmt.upper()} (rank-{d['rank']}):** `{Path(fp).name}`")
        if available_files:
            st.markdown("\n".join(available_files))
        else:
            st.warning("No output files generated")

# ─────────────────────────────────────────────────────────────────────────────
# Run button
# ─────────────────────────────────────────────────────────────────────────────

run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

if run_btn:
    if input_type == "SMILES" and not smiles_text:
        st.error("⚠️ Please enter a SMILES string")
    elif input_type == "SMI_FILE" and not uploaded:
        st.error("⚠️ Please upload a .smi file")
    elif input_type == "FILE" and not uploaded:
        st.error("⚠️ Please upload a PDB file")
    elif input_type == "FILE" and converted_smiles_from_pdb is None:
        st.error("⚠️ PDB → SMILES conversion failed. Cannot proceed.")
    elif not output_formats:
        st.error("⚠️ Please select at least one output format")
    else:
        try:
            with st.spinner("🔬 Running tautomer enumeration, microstate scoring & 3D generation…"):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)

                    if input_type == "FILE":
                        effective_input_type = "SMILES"
                        effective_smiles     = converted_smiles_from_pdb
                        effective_bytes      = None
                        effective_name       = None
                        st.info(f"🔄 Using canonical SMILES from PDB: `{converted_smiles_from_pdb}`")
                    elif input_type == "SMILES":
                        effective_input_type = "SMILES"
                        effective_smiles     = smiles_text
                        effective_bytes      = None
                        effective_name       = None
                    else:
                        effective_input_type = "SMI_FILE"
                        effective_smiles     = None
                        effective_bytes      = uploaded.read()
                        effective_name       = uploaded.name

                    out_dir = tmp / "out"
                    out = run_job(
                        input_type               = effective_input_type,
                        smiles_text              = effective_smiles,
                        uploaded_bytes           = effective_bytes,
                        uploaded_name            = effective_name,
                        target_pH                = target_pH,
                        output_name              = output_name,
                        out_dir                  = str(out_dir),
                        output_formats           = output_formats,
                        enumerate_stereoisomers  = enumerate_stereoisomers,
                        charge_mode              = "AUTO",    # legacy; not used in v4 pipeline
                        use_iupac_pka            = use_iupac_pka,
                        use_pubchem              = use_pubchem,
                        ph_window                = ph_window,
                        max_tautomers            = max_tautomers,
                        top_n_microstates        = top_n_microstates,
                        write_alt_3d_for_top_k   = write_alt_3d_for_top_k,
                    )

                    st.success("✅ Analysis complete!")

                    # ── Format / pKa warnings ─────────────────────────────────
                    if out.get("format_warnings"):
                        pka_warnings   = [w for w in out["format_warnings"] if "pKa prediction failed" in w]
                        info_warnings  = [w for w in out["format_warnings"] if w.startswith("ℹ️")]
                        other_warnings = [w for w in out["format_warnings"]
                                          if w not in pka_warnings and w not in info_warnings]
                        if pka_warnings:
                            with st.expander("⚠️ pKa Prediction Warnings", expanded=True):
                                for w in pka_warnings:
                                    st.warning(w)
                        if other_warnings:
                            with st.expander("⚠️ Other Warnings", expanded=False):
                                for w in other_warnings:
                                    st.warning(w)
                        for w in info_warnings:
                            st.info(w)

                    # ── Summary ───────────────────────────────────────────────
                    with st.expander("📊 Summary", expanded=True):
                        st.text(out["summary_text"])
                        total         = len(out["results"])
                        zwitterions   = sum(1 for r in out["results"] if r.get("is_zwitterion"))
                        ambiguous_cnt = sum(1 for r in out["results"] if r.get("ambiguous"))
                        info_parts    = []
                        if zwitterions:
                            info_parts.append(f"🧷 {zwitterions} zwitterion(s)")
                        if ambiguous_cnt:
                            info_parts.append(f"⚠️ {ambiguous_cnt} ambiguous assignment(s)")
                        if info_parts:
                            st.info(f"Generated {total} structure(s): {', '.join(info_parts)}")

                    # ── Per-ligand results ────────────────────────────────────
                    st.header("📈 Results")
                    results = out["results"]

                    if len(results) > 1:
                        tabs = st.tabs([r["name"] for r in results])
                        for tab, result in zip(tabs, results):
                            with tab:
                                display_ligand_result(
                                    result, out_dir, show_2d, show_3d, viewer_width, viewer_height
                                )
                    else:
                        display_ligand_result(
                            results[0], out_dir, show_2d, show_3d, viewer_width, viewer_height
                        )

                    # ── Downloads ─────────────────────────────────────────────
                    st.header("💾 Downloads")

                    log_file = out_dir / "processing.log"
                    if log_file.exists():
                        st.subheader("📋 Processing Log")
                        st.download_button(
                            "📄 Download Processing Log (.log)",
                            data=log_file.read_bytes(),
                            file_name="pkanet_v4_processing.log",
                            mime="text/plain",
                            use_container_width=True,
                            help="Tab-separated: Name | pH-SMILES | Charge | pKa | Source | Zwitterion | Ambiguous | PubChem_pKa",
                        )
                        st.markdown("---")

                    st.subheader("📦 Structure Files")
                    col1, col2 = st.columns(2)

                    with col1:
                        zip_all = tmp / "all_outputs.zip"
                        zip_all_outputs(str(out_dir), str(zip_all))
                        st.download_button(
                            "📦 Download ALL outputs (ZIP)",
                            data=zip_all.read_bytes(),
                            file_name="pkanet_v4_all_outputs.zip",
                            mime="application/zip",
                            use_container_width=True,
                            help="All structure files, logs, summaries, CSVs, and 2D images",
                        )

                    with col2:
                        zip_structures = tmp / "selected_structures.zip"
                        zip_minimized_structures(str(out_dir), str(zip_structures), output_formats)
                        btn_text = (
                            f"🧬 Download {output_formats[0]} files (ZIP)"
                            if len(output_formats) == 1
                            else f"🧬 Download {' + '.join(output_formats)} files (ZIP)"
                        )
                        st.download_button(
                            btn_text,
                            data=zip_structures.read_bytes(),
                            file_name=f"pkanet_v4_{'_'.join(f.lower() for f in output_formats)}.zip",
                            mime="application/zip",
                            use_container_width=True,
                            help=f"Only {' and '.join(output_formats)} structure files (SDF excluded)",
                        )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — About / Citation / Example
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About pKaNET Cloud v4")
st.sidebar.info("""
**pKaNET Cloud v4** uses:
- **Tautomer enumeration** (RDKit) + SMARTS-based plausibility scoring
- **Dimorphite-DL** for pH-dependent protonation state enumeration
- **Henderson–Hasselbalch** scoring for microstate ranking
- **PubChem** experimental pKa evidence (optional)
- **IUPAC pKa database** for high-confidence matches
- **pKaPredict** ML model as fallback
- **RDKit ETKDG + MMFF/UFF** for 3D structure generation

**New in v4:**
- Ranked microstate table with scores & flags
- Tautomer-aware protonation
- Ambiguity & borderline-pKa flags
- Amide/imidic-acid chemistry rules
- Per-atom charge reporting
""")

st.sidebar.markdown("### 📚 Citation")
st.sidebar.markdown("""
If you use this tool, please cite:
- DFDD project: Hengphasatporn K., Duan L., Harada R., Shigeta Y. JCIM (2026)
- Dimorphite-DL: Ropp PJ et al., J Cheminform (2019)
- pKaPredict: Floris A., Habert C., Baltruschat M., Czodrowski P. (2022)
""")

st.sidebar.markdown("### 💡 Examples")
st.sidebar.markdown("""
**Glycine** (zwitterion at pH 7.4):
```
NCC(=O)O
```
**Acylhydrazone** (amide NH preserved):
```
O=C(N/N=C/CC)C1=NC(C(N/N=C/CC)=O)=CC=C1
```
**Ibuprofen** (carboxylic acid):
```
CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
```
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    🧬 pKaNET Cloud v4 — Tautomer-aware microstate ranking |
    <a href='mailto:kowith@ccs.tsukuba.ac.jp'>kowith@ccs.tsukuba.ac.jp</a>
</div>
""", unsafe_allow_html=True)
