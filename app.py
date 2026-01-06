import streamlit as st
import tempfile
from pathlib import Path
from core import run_job, zip_all_outputs, zip_minimized_pdb_only
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem
import io

# Fix for RDKit Draw on headless servers
try:
    from rdkit.Chem import Draw
    DRAW_AVAILABLE = True
except (ImportError, OSError) as e:
    # RDKit Draw not available (missing X11 libraries on headless server)
    DRAW_AVAILABLE = False
    print(f"Warning: RDKit Draw not available: {e}")
    # Create a fallback Draw module
    class DrawFallback:
        @staticmethod
        def MolToImage(*args, **kwargs):
            return None
    Draw = DrawFallback()

st.set_page_config(page_title="pKaNET Cloud", layout="wide", page_icon="🧪")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stDownloadButton button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧪 pKaNET Cloud</div>', unsafe_allow_html=True)
#st.markdown('<div class="sub-header">AI-Based Protonation & 3D Structure Builder</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Machine-Learning–Driven Protonation & pH-Aware 3D Structure Generation<br>'
    '<span style="font-size:0.9em; font-weight:normal;">'
    'Instant pH-aware 3D structures for docking, virtual screening, and education — no setup required.'
    '</span>'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="sub-header">'
    'This is part of the <a href="https://github.com/nyelidl/DFDD" '
    'target="_blank"><strong>DFDD Project</strong></a>.'
    '</div>',
    unsafe_allow_html=True
)

# Sidebar configuration
st.sidebar.header("⚙️ Input / Options")
input_type = st.sidebar.selectbox("Input type", ["SMILES", "SMI_FILE", "FILE"])
target_pH = st.sidebar.slider("Target pH", 2.0, 12.0, 7.0, 0.1)
output_name = st.sidebar.text_input("Output name (for single SMILES/FILE)", value="ligand")

st.sidebar.header("📄 Output Format")
output_formats = st.sidebar.multiselect(
    "Select output formats",
    ["PDB", "MOL2"],
    default=["PDB"],
    help="SDF is always generated for 3D visualization"
)
if not output_formats:
    st.sidebar.warning("⚠️ Please select at least one output format")

# Add visualization options
st.sidebar.header("🎨 Visualization Options")
if DRAW_AVAILABLE:
    show_2d = st.sidebar.checkbox("Show 2D structure", value=True)
else:
    show_2d = False
    st.sidebar.info("ℹ️ 2D visualization not available on this server")
show_3d = st.sidebar.checkbox("Show 3D structure", value=True)

viewer_width = st.sidebar.slider("3D Viewer Width", 300, 800, 300, 50)
viewer_height = st.sidebar.slider("3D Viewer Height", 200, 600, 300, 50)

smiles_text = None
uploaded = None

# Input section
if input_type == "SMILES":
    smiles_text = st.text_area(
        "SMILES\nexample: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        height=120,
        placeholder="Paste a SMILES here:",
    )

elif input_type == "SMI_FILE":
    uploaded = st.file_uploader(
        "Upload .smi (SMILES [name] per line)",
        type=["smi", "txt"],
    )
    st.info("📝 Format: `SMILES [optional_name]` per line")

else:
    uploaded = st.file_uploader(
        "Upload ligand file",
        type=["pdb", "mol2", "sdf"],
    )
    st.info("📁 Supported formats: PDB, MOL2, SDF")


# Helper function for 2D visualization
def draw_molecule_2d(smiles_str, size=(400, 300)):
    """Generate 2D molecular structure image"""
    if not DRAW_AVAILABLE:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=size)
        return img
    except Exception as e:
        st.warning(f"Could not generate 2D structure: {e}")
        return None

# Helper function for 3D visualization
def create_3dmol_viewer(sdf_content, width=400, height=300):
    """Create py3Dmol viewer HTML - stick style with radius 0.2"""
    html_template = f"""
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
    return html_template

run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)


def display_ligand_result(result, out_dir, show_2d, show_3d, viewer_width, viewer_height):
    """Display results for a single ligand"""
    
    # Molecular information
    st.subheader("🔬 Molecular Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown(f"**Name:** `{result['name']}`")
        st.markdown(f"**Base SMILES:** `{result['base_smiles']}`")
        st.markdown(f"**pH-adjusted SMILES:** `{result['ph_smiles']}`")
    
    with info_col2:
        if result["pka_pred"] is not None:
            st.markdown(f"**Predicted pKa:** `{result['pka_pred']:.2f}`")
        st.markdown(f"**Formal Charge:** `{result['formal_charge']}`")
        st.markdown(f"**Target pH:** `{target_pH}`")
    
    # Visualization section
    if show_2d or show_3d:
        st.subheader("🎨 Structure Visualization")
        
        if show_2d and show_3d:
            viz_col1, viz_col2 = st.columns(2)
        else:
            viz_col1 = st.container()
            viz_col2 = None
        
        # 2D Structure
        if show_2d:
            with viz_col1:
                st.markdown("**2D Structure (pH-adjusted)**")
                if DRAW_AVAILABLE:
                    img_2d = draw_molecule_2d(result["ph_smiles"], size=(400, 300))
                    if img_2d:
                        st.image(img_2d, use_container_width=True)
                    else:
                        st.warning("Could not generate 2D structure")
                else:
                    st.info("2D visualization requires additional system libraries")
                    st.code(result["ph_smiles"], language=None)
        
        # 3D Structure - Always use SDF
        if show_3d:
            target_col = viz_col2 if viz_col2 else viz_col1
            with target_col:
                st.markdown("**🧊 3D Minimized Structure**")
                
                # Always use SDF for 3D visualization
                if "minimized_sdf" in result and result["minimized_sdf"]:
                    sdf_path = Path(result["minimized_sdf"])
                    if sdf_path.exists():
                        try:
                            with open(sdf_path, "r") as f:
                                sdf_data = f.read()
                            
                            # Escape special characters for JavaScript
                            sdf_data = sdf_data.replace('`', '\\`').replace('${', '\\${')
                            
                            viewer_html = create_3dmol_viewer(sdf_data, viewer_width, viewer_height)
                            components.html(viewer_html, height=viewer_height + 50, scrolling=False)
                            
                            st.caption("🖱️ Click and drag to rotate • Scroll to zoom")
                        except Exception as e:
                            st.warning(f"⚠️ 3D visualization failed: {e}")
                    else:
                        st.warning("3D structure file not found")
                else:
                    st.warning("SDF file not available for 3D visualization")
    
    # File information
    with st.expander("📁 Output Files"):
        available_files = []
        # Only show user-selected formats (not SDF, which is for visualization only)
        if "minimized_pdb" in result and result["minimized_pdb"]:
            if "PDB" in output_formats:
                available_files.append(f"- **PDB:** `{Path(result['minimized_pdb']).name}`")
        if "minimized_mol2" in result and result["minimized_mol2"]:
            if "MOL2" in output_formats:
                available_files.append(f"- **MOL2:** `{Path(result['minimized_mol2']).name}`")
        
        if available_files:
            st.markdown("\n".join(available_files))
            st.info("ℹ️ SDF files are generated automatically for 3D visualization")
        else:
            st.warning("No output files generated")


if run_btn:
    # Validation
    if input_type == "SMILES" and not smiles_text:
        st.error("⚠️ Please enter a SMILES string")
    elif input_type in ["SMI_FILE", "FILE"] and not uploaded:
        st.error("⚠️ Please upload a file")
    elif not output_formats:
        st.error("⚠️ Please select at least one output format")
    else:
        try:
            with st.spinner("🔬 Running pKa prediction and 3D generation..."):
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
                        output_formats=output_formats,
                    )

                    st.success("✅ Analysis complete!")
                    
                    # Display summary
                    with st.expander("📊 Summary", expanded=True):
                        st.text(out["summary_text"])
                    
                    # Display results for each ligand
                    st.header("📈 Results")
                    
                    results = out["results"]
                    
                    # Create tabs for multiple ligands or columns for single ligand
                    if len(results) > 1:
                        tabs = st.tabs([r["name"] for r in results])
                        
                        for tab, result in zip(tabs, results):
                            with tab:
                                display_ligand_result(result, out_dir, show_2d, show_3d, viewer_width, viewer_height)
                    else:
                        # Single ligand - use columns
                        result = results[0]
                        display_ligand_result(result, out_dir, show_2d, show_3d, viewer_width, viewer_height)
                    
                    # Download section
                    st.header("💾 Downloads")
                    
                    # Check if log file exists (for SMI_FILE input)
                    log_file = out_dir / "processing.log"
                    has_log = log_file.exists()
                    
                    if has_log:
                        # Show log file download first for SMI_FILE input
                        st.subheader("📋 Processing Log")
                        st.download_button(
                            "📄 Download Processing Log (.log)",
                            data=log_file.read_bytes(),
                            file_name="pkanet_processing.log",
                            mime="text/plain",
                            use_container_width=True,
                            help="Tab-separated file with: Name | pH-adjusted SMILES | Charge | pKa"
                        )
                        st.markdown("---")
                    
                    st.subheader("📦 Structure Files")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # ZIP everything
                        zip_all = tmp / "all_outputs.zip"
                        zip_all_outputs(str(out_dir), str(zip_all))
                        st.download_button(
                            "📦 Download ALL outputs (ZIP)",
                            data=zip_all.read_bytes(),
                            file_name="pkanet_outputs.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                    
                    with col2:
                        # ZIP only minimized PDB
                        zip_pdb = tmp / "minimized_pdb_only.zip"
                        zip_minimized_pdb_only(str(out_dir), str(zip_pdb))
                        st.download_button(
                            "🧬 Download minimized PDB only (ZIP)",
                            data=zip_pdb.read_bytes(),
                            file_name="minimized_pdb_outputs.zip",
                            mime="application/zip",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)


# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
**pKaNET Cloud** uses:
- **pKaPredict** for ML-based pKa prediction
- **Dimorphite-DL** for pH-dependent protonation
- **RDKit** for 3D structure generation
- **MMFF/UFF** for energy minimization
""")

st.sidebar.markdown("### 📚 Citation")
st.sidebar.markdown("""
If you use this tool, please cite:
- DFDD project: Hengphasatporn K et al., JCIM (2026)
- Dimorphite-DL: Ropp PJ et al., J Cheminform (2019)

We thank **Anastasia Floris, Candice Habert, Marcel Baltruschat, and Paul Czodrowski**
for developing **pKaPredict** and the study *“Machine Learning Meets pKa”*,
which inspired **pKaNET-Cloud**.

""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🧬 Developed for pH-dependent ligand preparation | 
    For questions: <a href='mailto:kowith@ccs.tsukuba.ac.jp'>kowith@ccs.tsukuba.ac.jp</a></p>
</div>
""", unsafe_allow_html=True)
