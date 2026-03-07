import streamlit as st
import tempfile
import subprocess
import shutil
import re
from pathlib import Path
from core import run_job, zip_all_outputs, zip_minimized_structures
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem
import io

# ── RDKit Draw (headless-safe) ─────────────────────────────────────────────
try:
    from rdkit.Chem import Draw
    DRAW_AVAILABLE = True
except (ImportError, OSError) as e:
    DRAW_AVAILABLE = False
    print(f"Warning: RDKit Draw not available: {e}")
    class DrawFallback:
        @staticmethod
        def MolToImage(*args, **kwargs): return None
    Draw = DrawFallback()

st.set_page_config(page_title="pKaNET Cloud", layout="wide", page_icon="🧪")

st.markdown("""
    <style>
    .main-header { font-size:2.5rem; font-weight:bold; color:#1f77b4;
                   text-align:center; margin-bottom:0.5rem; }
    .sub-header  { text-align:center; color:#666; margin-bottom:2rem; }
    .result-card { background-color:#f0f2f6; padding:1rem;
                   border-radius:0.5rem; margin:1rem 0; }
    .stDownloadButton button { width:100%; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧪 pKaNET Cloud</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Machine-Learning–Driven Protonation & pH-Aware 3D Structure Generation<br>'
    '<span style="font-size:0.9em;">'
    'Instant pH-aware 3D structures for docking, virtual screening, and education – '
    'with automatic R/S stereoisomer enumeration and zwitterion control.'
    '</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">This is part of the '
    '<a href="https://github.com/nyelidl/DFDD" target="_blank"><strong>DFDD Project</strong></a>.'
    '</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# xTB pKa helpers
# ═══════════════════════════════════════════════════════════════════════════
HARTREE_TO_KCAL = 627.51
RT_LN10         = 1.36   # kcal/mol at 298 K

XTB_REFERENCES = {
    "amine": {
        "HA_smi":    "CC[NH3+]", "A_smi":  "CCN",
        "chrg_HA":   "+1",        "chrg_A": "0",
        "pKa_ref":   10.7,
        "label":     "Amine (ref: ethylamine, pKa 10.7)",
        "rxn":       None,          # handled by _protonate_amine() — avoids valence error
        "direction": "protonate",
    },
    "acid": {
        "HA_smi":    "CC(=O)O",  "A_smi":  "CC(=O)[O-]",
        "chrg_HA":   "0",         "chrg_A": "-1",
        "pKa_ref":   4.75,
        "label":     "Carboxylic acid (ref: acetic acid, pKa 4.75)",
        "rxn":       "[CX3](=O)[OX2H1:1]>>[CX3](=O)[OX1-:1]",
        "direction": "deprotonate",
    },
    "phenol": {
        "HA_smi":    "Oc1ccccc1", "A_smi": "[O-]c1ccccc1",
        "chrg_HA":   "0",          "chrg_A": "-1",
        "pKa_ref":   9.99,
        "label":     "Phenol (ref: phenol, pKa 9.99)",
        "rxn":       "[OX2H][c:1]>>[OX1-][c:1]",
        "direction": "deprotonate",
    },
}

def _xtb_available():
    return shutil.which("xtb") is not None

def _write_xyz(mol, file_name):
    n       = mol.GetNumAtoms()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    conf    = mol.GetConformers()[0]
    with open(file_name, "w") as f:
        f.write(f"{n}\ntitle\n")
        for i, sym in enumerate(symbols):
            p = conf.GetAtomPosition(i)
            f.write(f"{sym} {p.x:.6f} {p.y:.6f} {p.z:.6f}\n")

def _get_best_conf(mol, n_confs=20):
    new_mol = Chem.Mol(mol)
    AllChem.EmbedMultipleConfs(
        mol, numConfs=n_confs,
        useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
    energies = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
    min_idx  = min(range(len(energies)), key=lambda i: energies[i][1])
    new_mol.AddConformer(mol.GetConformer(min_idx))
    return new_mol

def _run_xtb_calc(mol, flags, xyz_name):
    mol = Chem.AddHs(mol)
    mol = _get_best_conf(mol)
    _write_xyz(mol, xyz_name)
    result = subprocess.run(
        ["bash", "-lc", f"xtb {xyz_name} {flags}"],
        capture_output=True, text=True)
    return result.stdout + result.stderr

def _get_energy(output):
    m = re.search(r"TOTAL ENERGY\s+([-\d.]+)\s+Eh", output)
    if not m:
        raise RuntimeError(f"Cannot parse xTB energy.\n{output[-800:]}")
    return float(m.group(1))

def _run_rxn(mol, smarts):
    rxn  = AllChem.ReactionFromSmarts(smarts)
    ps   = rxn.RunReactants((mol,))
    if not ps:
        raise ValueError(f"Reaction produced no products: {smarts}")
    prod = ps[0][0]
    Chem.SanitizeMol(prod)
    return prod


def _protonate_amine(mol):
    """Safely add exactly one proton to the first free amine nitrogen.

    Works for primary (R-NH2 → R-NH3+), secondary (R2-NH → R2-NH2+), and
    tertiary (R3-N → R3-NH+) amines without hitting the valence error that
    the hard-coded [NH3+] SMARTS causes on secondary/tertiary centres.
    """
    from rdkit.Chem import RWMol
    amide_pat = Chem.MolFromSmarts("[N;$(NC=O)]")
    rwmol = RWMol(Chem.AddHs(mol))
    for atom in rwmol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        if atom.GetFormalCharge() != 0:
            continue
        # Skip amide nitrogens
        if mol.HasSubstructMatch(amide_pat):
            amide_idxs = {m[0] for m in mol.GetSubstructMatches(amide_pat)}
            if atom.GetIdx() in amide_idxs:
                continue
        total_h = atom.GetTotalNumHs()
        if total_h == 0:
            continue
        # Add exactly one proton
        atom.SetFormalCharge(1)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(total_h + 1)
        try:
            Chem.SanitizeMol(rwmol)
            return Chem.RemoveHs(rwmol.GetMol())
        except Exception:
            # Roll back and try the next nitrogen
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(False)
            atom.SetNumExplicitHs(total_h)
    raise ValueError("No protonatable amine nitrogen found")

def _detect_ionizable_groups(mol):
    patterns = {
        "amine":  "[NX3;H1,H2;!$(NC=O)]",
        "acid":   "[CX3](=O)[OX2H1]",
        "phenol": "[OX2H][c]",
    }
    found = [g for g, p in patterns.items()
             if mol.HasSubstructMatch(Chem.MolFromSmarts(p))]
    return found if found else ["unknown"]

def run_xtb_pka(smiles_str: str, tmp_dir: Path) -> list[dict]:
    """Run xTB isodesmic pKa. Returns list of {group, label, pKa, dE_kcal, dpKa, error}."""
    mol = Chem.MolFromSmiles(smiles_str.strip())
    if mol is None:
        return [{"group": "error", "label": "Parse error", "pKa": None,
                 "error": "Cannot parse SMILES"}]

    groups  = _detect_ionizable_groups(mol)
    results = []

    for group in groups:
        if group == "unknown":
            results.append({
                "group": "unknown", "label": "No ionizable group detected",
                "pKa": None, "error": "No amine / carboxylic acid / phenol found",
            })
            continue

        ref = XTB_REFERENCES[group]
        try:
            E_HAref = _get_energy(_run_xtb_calc(
                Chem.MolFromSmiles(ref["HA_smi"]),
                f"--opt --alpb water --chrg {ref['chrg_HA']}",
                str(tmp_dir / "HAref.xyz")))
            E_Aref  = _get_energy(_run_xtb_calc(
                Chem.MolFromSmiles(ref["A_smi"]),
                f"--opt --alpb water --chrg {ref['chrg_A']}",
                str(tmp_dir / "Aref.xyz")))

            if ref["direction"] == "protonate":
                # Use safe per-atom protonation — avoids valence errors on
                # secondary/tertiary amines that [NH3+] SMARTS cannot handle
                HA_guest = _protonate_amine(mol)
                A_guest  = mol
                chrg_HA, chrg_A = "+1", "0"
            else:
                HA_guest = mol
                A_guest  = _run_rxn(mol, ref["rxn"])
                chrg_HA, chrg_A = "0", "-1"

            E_HA = _get_energy(_run_xtb_calc(
                HA_guest, f"--opt --alpb water --chrg {chrg_HA}",
                str(tmp_dir / "HA_guest.xyz")))
            E_A  = _get_energy(_run_xtb_calc(
                A_guest, f"--opt --alpb water --chrg {chrg_A}",
                str(tmp_dir / "A_guest.xyz")))

            dE_kcal = ((E_HAref + E_A) - (E_Aref + E_HA)) * HARTREE_TO_KCAL
            dpKa    = dE_kcal / RT_LN10
            pKa_xtb = ref["pKa_ref"] + dpKa

            results.append({
                "group":   group,
                "label":   ref["label"],
                "pKa":     pKa_xtb,
                "dE_kcal": dE_kcal,
                "dpKa":    dpKa,
                "error":   None,
            })

        except Exception as e:
            results.append({
                "group": group, "label": ref["label"],
                "pKa": None, "error": str(e),
            })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PDB → SMILES conversion
# ═══════════════════════════════════════════════════════════════════════════
def pdb_to_canonical_smiles(pdb_bytes: bytes) -> tuple[str | None, str | None]:
    if shutil.which("obabel") is None:
        return None, (
            "Open Babel (obabel) is not installed or not found in PATH. "
            "Please install it: conda install -c conda-forge openbabel"
        )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp      = Path(tmp)
            pdb_file = tmp / "input.pdb"
            smi_file = tmp / "output.smi"
            pdb_file.write_bytes(pdb_bytes)
            result = subprocess.run(
                ["obabel", str(pdb_file), "-O", str(smi_file), "--canonical"],
                capture_output=True, text=True, timeout=30)
            if not smi_file.exists() or smi_file.stat().st_size == 0:
                return None, f"obabel produced no output. stderr: {result.stderr.strip() or '(none)'}"
            raw    = smi_file.read_text(encoding="utf-8", errors="replace").strip()
            if not raw:
                return None, "obabel produced an empty SMILES file."
            smiles = raw.splitlines()[0].split()[0].strip()
            if not smiles:
                return None, "Could not parse SMILES from obabel output."
            return smiles, None
    except subprocess.TimeoutExpired:
        return None, "obabel conversion timed out (>30 s)."
    except Exception as exc:
        return None, f"Unexpected error during PDB→SMILES conversion: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Input / Options")
input_type  = st.sidebar.selectbox("Input type", ["SMILES", "SMI_FILE", "FILE"])
target_pH   = st.sidebar.slider("Target pH", 2.0, 12.0, 7.4, 0.1)
output_name = st.sidebar.text_input("Output name (for single SMILES/FILE)", value="ligand")

st.sidebar.header("🧬 Stereochemistry")
enumerate_stereoisomers = st.sidebar.checkbox(
    "Enumerate R/S stereoisomers", value=True,
    help="Automatically generate both R and S stereoisomers for undefined chiral centers.")

st.sidebar.header("⚡ Charge Mode")
charge_mode = st.sidebar.selectbox(
    "Protonation state selection",
    ["AUTO", "FORCE_ZWITTERION", "NORMAL"], index=0,
    help=(
        "AUTO: Use Dimorphite-DL dominant microspecies (first variant)\n"
        "FORCE_ZWITTERION: Return strict zwitterion if present; else most neutral\n"
        "NORMAL: Choose most neutral state (smallest |net charge|)"
    ))
if charge_mode == "FORCE_ZWITTERION":
    st.sidebar.info(
        "🧷 **Zwitterion mode**: Will prioritize structures with "
        "both positive and negative atoms and net charge = 0")

# ── pKa Prediction options ─────────────────────────────────────────────────
st.sidebar.header("🔬 pKa Prediction")

# IUPAC is always attempted first — both methods share this behaviour
use_iupac_pka = True

# Always show the IUPAC-first note
st.sidebar.markdown(
    "🗄️ **IUPAC pKa database** is always checked first.  \n"
    "The toggle below selects the **fallback method** used when no IUPAC match is found.")

xtb_available = _xtb_available()

# Toggle label changes to reflect xTB availability
_toggle_label = (
    "Fallback: pKaPredict ML  ·  xTB GFN2/ALPB ⚛️"
    if xtb_available
    else "Fallback: pKaPredict ML  ·  xTB ⚠️ (not installed)"
)

# st.toggle: False = pKaPredict ML, True = xTB
_xtb_toggle = st.sidebar.toggle(
    _toggle_label,
    value=False,
    disabled=not xtb_available,
    help=(
        "**OFF → pKaPredict ML**\n"
        "IUPAC database checked first; if no match, the ML model predicts pKa.\n\n"
        "**ON → xTB GFN2/ALPB(water)**\n"
        "IUPAC database checked first; if matched, IUPAC pKa is shown alongside xTB. "
        "If no IUPAC match, xTB isodesmic pKa is the sole source (ML is skipped). "
        "Supports amine / carboxylic acid / phenol groups. Accuracy ±1–2 pKa units."
        if xtb_available
        else "**xTB is not available** — install xTB to enable quantum-chemical pKa."
    ),
)

use_xtb_pka = _xtb_toggle and xtb_available

# Active strategy caption
if use_xtb_pka:
    st.sidebar.caption(
        "⚛️ Active: IUPAC → xTB GFN2/ALPB  |  no IUPAC match → xTB only (ML skipped)")
else:
    st.sidebar.caption(
        "🤖 Active: IUPAC → pKaPredict ML  |  no IUPAC match → ML only")

if not xtb_available:
    st.sidebar.caption("⚠️ xTB binary not found in PATH — toggle disabled.")

st.sidebar.header("📄 Output Format")
output_formats = st.sidebar.multiselect(
    "Select output formats", ["PDB", "MOL2"], default=["PDB"],
    help="SDF is always generated for 3D visualization")
if not output_formats:
    st.sidebar.warning("⚠️ Please select at least one output format")

st.sidebar.header("🎨 Visualization Options")
if DRAW_AVAILABLE:
    show_2d = st.sidebar.checkbox("Show 2D structure", value=True)
else:
    show_2d = False
    st.sidebar.info("ℹ️ 2D visualization not available on this server")
show_3d       = st.sidebar.checkbox("Show 3D structure", value=True)
viewer_width  = st.sidebar.slider("3D Viewer Width",  300, 800, 300, 50)
viewer_height = st.sidebar.slider("3D Viewer Height", 200, 600, 300, 50)


# ═══════════════════════════════════════════════════════════════════════════
# Input widgets
# ═══════════════════════════════════════════════════════════════════════════
smiles_text = None
uploaded    = None

if input_type == "SMILES":
    smiles_text = st.text_area(
        "SMILES\nexample: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", height=120,
        placeholder="Paste a SMILES (RDKit-canonical SMILES) here:")
elif input_type == "SMI_FILE":
    uploaded = st.file_uploader("Upload .smi (SMILES [name] per line)", type=["smi", "txt"])
    st.info("📝 Format: `SMILES [optional_name]` per line")
else:
    uploaded = st.file_uploader("Upload ligand PDB file", type=["pdb"])
    st.info("📝 The PDB file will be automatically converted to canonical SMILES using Open Babel.")

# ── Live PDB → SMILES preview ──────────────────────────────────────────────
converted_smiles_from_pdb: str | None = None

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


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════
def draw_molecule_2d(smiles_str, size=(400, 300)):
    if not DRAW_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None: return None
        AllChem.Compute2DCoords(mol)
        return Draw.MolToImage(mol, size=size)
    except Exception as e:
        st.warning(f"Could not generate 2D structure: {e}")
        return None


def create_3dmol_viewer(sdf_content, width=400, height=300):
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


def display_ligand_result(result, out_dir, show_2d, show_3d,
                          viewer_width, viewer_height,
                          xtb_results=None):
    """Display results for a single ligand."""

    st.subheader("🔬 Molecular Information")
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown(f"**Name:** `{result['name']}`")
        st.markdown(f"**Base SMILES:** `{result['base_smiles']}`")
        st.markdown(f"**pH-adjusted SMILES:** `{result['ph_smiles']}`")

    with info_col2:
        st.markdown(f"**Target pH:** `{target_pH}`")

        # ── pKa display — implements the full logic matrix ─────────────
        iupac_matched = result.get("pka_iupac_matched", False)
        pka_source    = result.get("pka_source", "")
        xtb_pkas      = [r for r in (xtb_results or []) if r.get("pKa") is not None]

        if iupac_matched and result["pka_pred"] is not None:
            # IUPAC matched — always show it
            st.markdown(
                f"**Predicted pKa ({pka_source}):** `{result['pka_pred']:.2f}`")
            if xtb_pkas:
                # xTB also enabled — show alongside
                for xr in xtb_pkas:
                    st.markdown(
                        f"**xTB pKa ({xr['group']}):** `{xr['pKa']:.1f}` "
                        f"*(ΔE = {xr['dE_kcal']:+.3f} kcal/mol)*")

        elif pka_source == "xTB" or (use_xtb_pka and not iupac_matched):
            # No IUPAC match + xTB enabled → xTB only, skip ML
            if xtb_pkas:
                for xr in xtb_pkas:
                    st.markdown(
                        f"**xTB pKa ({xr['group']}):** `{xr['pKa']:.1f}` "
                        f"*(ΔE = {xr['dE_kcal']:+.3f} kcal/mol)*")
                st.caption(
                    "⚛️ No IUPAC match — xTB GFN2/ALPB(water) used as sole pKa source")
            else:
                st.markdown("**Predicted pKa:** `xTB calculation pending or failed`")
                st.caption("⚛️ No IUPAC match; ML skipped because xTB is enabled")

        elif result["pka_pred"] is not None:
            # ML only (IUPAC off or no match, xTB off)
            st.markdown(
                f"**Predicted pKa ({pka_source}):** `{result['pka_pred']:.2f}`")

        else:
            # All failed
            st.markdown("**Predicted pKa:** `N/A` ⚠️")
            st.caption("⚠️ pKa prediction unavailable — check warnings below")

        # ── Show xTB errors if any ─────────────────────────────────────
        if xtb_results:
            xtb_errors = [r for r in xtb_results if r.get("error") and r["group"] != "unknown"]
            for xe in xtb_errors:
                st.caption(f"⚠️ xTB ({xe['group']}): {xe['error']}")

        # ── Charge & zwitterion info ───────────────────────────────────
        st.markdown(f"**Net Formal Charge at pH {target_pH}:** `{result['formal_charge']:+d}`")

        if result.get("is_zwitterion", False):
            st.markdown("**Zwitterion (strict):** `YES` 🧷")
            st.caption(
                f"✓ Has {result.get('n_pos_atoms',0)} positive and "
                f"{result.get('n_neg_atoms',0)} negative atoms, net charge = 0")
        else:
            st.markdown("**Zwitterion (strict):** `NO`")
            if result.get("has_pos") and result.get("has_neg"):
                st.caption(
                    f"Has {result.get('n_pos_atoms',0)} positive and "
                    f"{result.get('n_neg_atoms',0)} negative atoms, but net charge ≠ 0")

        if "stereoisomer_id" in result:
            st.markdown(f"**Stereoisomer:** `{result['stereoisomer_id']}`")

    # ── Visualization ──────────────────────────────────────────────────────
    if show_2d or show_3d:
        st.subheader("🎨 Structure Visualization")

        if show_2d and show_3d:
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                st.markdown("**2D Structure**")
                if DRAW_AVAILABLE:
                    img = draw_molecule_2d(result["ph_smiles"], size=(400, 300))
                    if img: st.image(img, use_container_width=True)
                    else:   st.warning("Could not generate 2D structure")
            with viz_col2:
                st.markdown("**3D Structure**")
                _show_3d_viewer(result, viewer_width, viewer_height)

        elif show_2d:
            st.markdown("**2D Structure**")
            if DRAW_AVAILABLE:
                img = draw_molecule_2d(result["ph_smiles"], size=(600, 400))
                if img: st.image(img, use_container_width=True)

        elif show_3d:
            st.markdown("**3D Structure**")
            _show_3d_viewer(result, viewer_width, viewer_height)

    # ── Output files expander ──────────────────────────────────────────────
    with st.expander("📁 Output Files"):
        available_files = []
        for key, label in [
            ("minimized_pdb",  "PDB"),
            ("minimized_mol2", "MOL2"),
            ("minimized_sdf",  "SDF"),
        ]:
            if result.get(key) and Path(result[key]).exists():
                available_files.append(f"- **{label}:** `{Path(result[key]).name}`")
        if available_files:
            st.markdown("\n".join(available_files))
        else:
            st.warning("No output files generated")


def _show_3d_viewer(result, width, height):
    if "minimized_sdf" in result:
        sdf_path = Path(result["minimized_sdf"])
        if sdf_path.exists():
            try:
                viewer_html = create_3dmol_viewer(
                    sdf_path.read_text(), width=width, height=height)
                components.html(viewer_html, height=height + 20, scrolling=False)
            except Exception as e:
                st.warning(f"⚠️ 3D visualization failed: {e}")
        else:
            st.warning("3D structure file not found")
    else:
        st.warning("SDF file not available for 3D visualization")


# ═══════════════════════════════════════════════════════════════════════════
# Run button
# ═══════════════════════════════════════════════════════════════════════════
run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

if run_btn:
    # Validation
    if   input_type == "SMILES"   and not smiles_text:
        st.error("⚠️ Please enter a SMILES string")
    elif input_type == "SMI_FILE" and not uploaded:
        st.error("⚠️ Please upload a .smi file")
    elif input_type == "FILE"     and not uploaded:
        st.error("⚠️ Please upload a PDB file")
    elif input_type == "FILE"     and converted_smiles_from_pdb is None:
        st.error("⚠️ PDB → SMILES conversion failed. Cannot proceed.")
    elif not output_formats:
        st.error("⚠️ Please select at least one output format")
    else:
        try:
            with st.spinner("🔬 Running pKa prediction and 3D generation..."):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)

                    # Resolve effective input
                    if input_type == "FILE":
                        effective_input_type = "SMILES"
                        effective_smiles     = converted_smiles_from_pdb
                        effective_bytes      = None
                        effective_name       = None
                        st.info(f"🔄 Using canonical SMILES derived from PDB: "
                                f"`{converted_smiles_from_pdb}`")
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

                    # ── Main job (pKa / protonation / 3D) ────────────────
                    out = run_job(
                        input_type              = effective_input_type,
                        smiles_text             = effective_smiles,
                        uploaded_bytes          = effective_bytes,
                        uploaded_name           = effective_name,
                        target_pH               = target_pH,
                        output_name             = output_name,
                        out_dir                 = str(out_dir),
                        output_formats          = output_formats,
                        enumerate_stereoisomers = enumerate_stereoisomers,
                        charge_mode             = charge_mode,
                        use_iupac_pka           = use_iupac_pka,
                        use_xtb_pka             = use_xtb_pka,
                    )

                    # ── xTB pKa (runs after run_job, once per unique base SMILES) ──
                    xtb_results_map: dict[str, list[dict]] = {}
                    if use_xtb_pka:
                        with st.spinner("⚛️ Running xTB pKa calculations..."):
                            xtb_tmp = tmp / "xtb_work"
                            xtb_tmp.mkdir(exist_ok=True)
                            seen = set()
                            for r in out["results"]:
                                base_smi = r.get("base_smiles", "")
                                if base_smi and base_smi not in seen:
                                    seen.add(base_smi)
                                    try:
                                        xtb_results_map[base_smi] = run_xtb_pka(
                                            base_smi, xtb_tmp)
                                    except Exception as e:
                                        xtb_results_map[base_smi] = [{
                                            "group": "error", "label": "xTB error",
                                            "pKa": None, "error": str(e),
                                        }]

                        # Append xTB results to summary.txt
                        summary_path = out_dir / "summary.txt"
                        with open(summary_path, "a") as f:
                            f.write("\n" + "=" * 80 + "\n")
                            f.write("xTB pKa Results (GFN2 / ALPB water)\n")
                            f.write("Method: isodesmic proton transfer | "
                                    "Accuracy: ±1–2 pKa units\n")
                            f.write("=" * 80 + "\n")
                            for base_smi, xtb_res in xtb_results_map.items():
                                f.write(f"\nSMILES: {base_smi}\n")
                                for xr in xtb_res:
                                    if xr.get("pKa") is not None:
                                        f.write(
                                            f"  xTB pKa ({xr['group']:8s}) : "
                                            f"{xr['pKa']:.1f}  "
                                            f"[ΔE={xr['dE_kcal']:+.3f} kcal/mol]\n")
                                    elif xr.get("error"):
                                        f.write(
                                            f"  xTB pKa ({xr.get('group','?'):8s}) : "
                                            f"ERROR — {xr['error']}\n")

                        # Also update processing.log with xTB column (SMI_FILE mode)
                        log_path = out_dir / "processing.log"
                        if log_path.exists():
                            lines     = log_path.read_text().splitlines()
                            new_lines = []
                            for line in lines:
                                if line.startswith("#") or not line.strip():
                                    new_lines.append(line)
                                    continue
                                parts    = line.split("\t")
                                base_smi = None
                                # Match by pH-adjusted SMILES back to base SMILES
                                for r in out["results"]:
                                    if len(parts) > 1 and parts[1] == r.get("ph_smiles"):
                                        base_smi = r.get("base_smiles")
                                        break
                                xtb_col = "-"
                                if base_smi and base_smi in xtb_results_map:
                                    xtb_parts = [
                                        f"{xr['group']}:{xr['pKa']:.1f}"
                                        for xr in xtb_results_map[base_smi]
                                        if xr.get("pKa") is not None
                                    ]
                                    if xtb_parts:
                                        xtb_col = ";".join(xtb_parts)
                                # Replace existing last column or append
                                if len(parts) == 7:
                                    parts[6]  = xtb_col
                                    new_lines.append("\t".join(parts))
                                else:
                                    new_lines.append(line + f"\t{xtb_col}")
                            log_path.write_text("\n".join(new_lines) + "\n")

                    st.success("✅ Analysis complete!")

                    # ── Warnings ──────────────────────────────────────────
                    if out.get("format_warnings"):
                        pka_warnings   = [w for w in out["format_warnings"]
                                          if "pKa prediction failed" in w]
                        info_warnings  = [w for w in out["format_warnings"]
                                          if w.startswith("ℹ️")]
                        other_warnings = [w for w in out["format_warnings"]
                                          if w not in pka_warnings
                                          and w not in info_warnings]
                        if pka_warnings:
                            with st.expander("⚠️ pKa Prediction Warnings", expanded=True):
                                for w in pka_warnings: st.warning(w)
                                st.info(
                                    "💡 pH-adjusted structure and formal charge are still "
                                    "calculated correctly using Dimorphite-DL.")
                        if other_warnings:
                            with st.expander("⚠️ Format Warnings", expanded=False):
                                for w in other_warnings: st.warning(w)
                        for w in info_warnings: st.info(w)

                    # ── Summary ───────────────────────────────────────────
                    with st.expander("📊 Summary", expanded=True):
                        st.text(out["summary_text"])
                        total         = len(out["results"])
                        zwitterions   = sum(1 for r in out["results"]
                                            if r.get("is_zwitterion", False))
                        stereoisomers = len(set(
                            r.get("stereoisomer_id") for r in out["results"]
                            if "stereoisomer_id" in r))
                        info_parts = []
                        if enumerate_stereoisomers and stereoisomers > 0:
                            info_parts.append(f"🧬 {stereoisomers} stereoisomer type(s)")
                        if zwitterions > 0:
                            info_parts.append(f"🧷 {zwitterions} zwitterion(s) (strict)")
                        if info_parts:
                            st.info(f"Generated {total} total structure(s): "
                                    f"{', '.join(info_parts)}")

                    # ── Per-ligand results ─────────────────────────────────
                    st.header("📈 Results")
                    results = out["results"]

                    if len(results) > 1:
                        tabs = st.tabs([r["name"] for r in results])
                        for tab, result in zip(tabs, results):
                            with tab:
                                xtb_res = (xtb_results_map.get(result.get("base_smiles", ""))
                                           if use_xtb_pka else None)
                                display_ligand_result(
                                    result, out_dir, show_2d, show_3d,
                                    viewer_width, viewer_height,
                                    xtb_results=xtb_res)
                    else:
                        xtb_res = (xtb_results_map.get(results[0].get("base_smiles", ""))
                                   if use_xtb_pka else None)
                        display_ligand_result(
                            results[0], out_dir, show_2d, show_3d,
                            viewer_width, viewer_height,
                            xtb_results=xtb_res)

                    # ── Downloads ──────────────────────────────────────────
                    st.header("💾 Downloads")

                    log_file = out_dir / "processing.log"
                    if log_file.exists():
                        st.subheader("📋 Processing Log")
                        st.download_button(
                            "📄 Download Processing Log (.log)",
                            data=log_file.read_bytes(),
                            file_name="pkanet_processing.log",
                            mime="text/plain",
                            use_container_width=True,
                            help="Tab-separated: Name | pH-SMILES | Charge | "
                                 "pKa | Source | Zwitterion | xTB pKa")
                        st.markdown("---")

                    st.subheader("📦 Structure Files")
                    col1, col2 = st.columns(2)
                    with col1:
                        zip_all = tmp / "all_outputs.zip"
                        zip_all_outputs(str(out_dir), str(zip_all))
                        st.download_button(
                            "📦 Download ALL outputs (ZIP)",
                            data=zip_all.read_bytes(),
                            file_name="pkanet_all_outputs.zip",
                            mime="application/zip",
                            use_container_width=True)
                    with col2:
                        zip_structures = tmp / "selected_structures.zip"
                        zip_minimized_structures(
                            str(out_dir), str(zip_structures), output_formats)
                        btn_text = (
                            f"🧬 Download {output_formats[0]} files (ZIP)"
                            if len(output_formats) == 1
                            else f"🧬 Download {' + '.join(output_formats)} files (ZIP)")
                        st.download_button(
                            btn_text,
                            data=zip_structures.read_bytes(),
                            file_name=f"pkanet_{'_'.join(f.lower() for f in output_formats)}.zip",
                            mime="application/zip",
                            use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — About / Citation / Example
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
**pKaNET Cloud** uses:
- **IUPAC pKa database** — always checked first
- **pKaPredict (ML)** — fallback when no IUPAC match *(toggle OFF)*
- **xTB GFN2/ALPB(water)** — quantum-chemical fallback *(toggle ON)*
- **Dimorphite-DL** for pH-dependent protonation
- **RDKit** for 3D structure generation & MMFF/UFF minimization

**pKa strategy (IUPAC always first):**
- Toggle OFF → IUPAC if matched, else **pKaPredict ML**
- Toggle ON  → IUPAC if matched + **xTB** alongside;
  no IUPAC match → **xTB only** (ML skipped)

**Charge Modes:**
- **AUTO**: Dimorphite-DL dominant microspecies
- **FORCE_ZWITTERION**: Prioritize strict zwitterions
- **NORMAL**: Most neutral state
""")

st.sidebar.markdown("### 📚 Citation")
st.sidebar.markdown("""
If you use this tool, please cite:
- DFDD project: Hengphasatporn K., Duan L., Harada R., Shigeta Y. JCIM (2026)
- Dimorphite-DL: Ropp PJ et al., J Cheminform (2019)

We thank **Anastasia Floris, Candice Habert, Marcel Baltruschat, and Paul Czodrowski**
for developing **pKaPredict** and the study *"Machine Learning Meets pKa"*.
""")

st.sidebar.markdown("### 💡 Example")
st.sidebar.markdown("""
Try **Glycine** (simplest amino acid):
```
C(C(=O)O)N
```
Or **Erlotinib** (EGFR inhibitor):
```
COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC
```
Use **FORCE_ZWITTERION** mode for zwitterionic forms!
""")

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;font-size:0.9rem;'>
    <p>🧬 Developed for pH-dependent ligand preparation |
    For questions:
    <a href='mailto:kowith@ccs.tsukuba.ac.jp'>kowith@ccs.tsukuba.ac.jp</a></p>
</div>
""", unsafe_allow_html=True)
