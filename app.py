import streamlit as st
import tempfile
import subprocess
import shutil
import io
import re
import contextlib
import requests as _requests
from urllib.parse import quote as _url_quote
from pathlib import Path
from core import (
    run_job, zip_all_outputs, zip_minimized_structures, DISPLAY_COLS, _PKA_BACKEND,
    __version__ as _PKANET_VERSION,
    _PYMUPDF_OK,
    pdf2smi_get_page_count, pdf2smi_render_page, pdf2smi_extract_text,
    pdf2smi_build_all, pdf2smi_make_grid_image, pdf2smi_to_smi_bytes, pdf2smi_to_csv_bytes,
    make_structure_grid_image, parse_smi_lines,
)

try:
    import struct2smi
    _STRUCT2SMI_OK = True
except Exception:
    _STRUCT2SMI_OK = False
import pandas as pd
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()  # lets Image.open() handle .heic/.heif transparently
    _HEIF_OK = True
except ImportError:
    _HEIF_OK = False
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

_LOGO_URL = (
    "https://raw.githubusercontent.com/nyelidl/pKaNET_Cloud"
    "/30b7d67ba323099789fbb4f4e597cb7ab9f8495d/pKaN2.svg"
)
_HEADER_LOGO_URL = (
    "https://raw.githubusercontent.com/nyelidl/pKaNET_Cloud"
    "/2b736479e65b589f1395ed697c17a0ccbeb60d6e/pKaN.svg"
)
# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="pKaNET Cloud+",
    layout="wide",
    page_icon="https://raw.githubusercontent.com/nyelidl/pKaNET_Cloud/9cbb6681fbceb408e83e428ae44dd6619a8a4ec0/pKaN2.svg"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@900&display=swap');

.main-header {
    font-family: 'Arial Rounded MT Bold', 'Arial Rounded MT', 'Nunito', sans-serif;
    font-size: 2.5rem;
    font-weight: 900;
    text-align: center;
    margin-bottom: 0.5rem;
    background: linear-gradient(120deg, #00f5c4 0%, #60b4ff 40%, #ff6eb4 80%, #ffe660 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: headerShimmer 4s linear infinite;
    display: inline-block;
    width: 100%;
}
@keyframes headerShimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.main-header-wrap { text-align: center; }
.main-header-logo {
    width: 80px;
    height: 80px;
    vertical-align: middle;
    margin-right: 0.4rem;
    margin-bottom: 0.3rem;
    filter: drop-shadow(0 0 10px rgba(0,245,196,.45));
}
.sub-header  { text-align:center; color:#666; margin-bottom:2rem; }
.stDownloadButton button { width:100%; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    f'<div class="main-header-wrap">'
    f'<img class="main-header-logo" src="{_HEADER_LOGO_URL}" alt="pKaNET logo" />'
    f'<span class="main-header">pKaNET Cloud+</span>'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">'
    f'pKaNET v{_PKANET_VERSION} · Tautomer-aware microstate ranking · Henderson–Hasselbalch scoring · '
    'PubChem pKa evidence · pH-adjusted 3D structures<br>'
    'Part of the <a href="https://github.com/nyelidl/DFDD" target="_blank">'
    '<strong>DFDD Project</strong></a>'
    '</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Name / SMILES resolution (port from Guest Preparation notebook)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_name(name: str) -> str:
    name = (name or "").strip() or "mol"
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name


def _is_smiles(text: str) -> bool:
    """Heuristic: does this string look like SMILES rather than a compound name?"""
    text = text.strip().split()[0] if text.strip() else ""
    if not text:
        return False
    smi_chars = set("CNOPSFIBrClcnopsb[]()=\\/#@+\\-0123456789%.")
    ratio = sum(1 for c in text if c in smi_chars) / len(text)
    if ratio > 0.70:
        return True
    with contextlib.redirect_stderr(io.StringIO()):
        mol = Chem.MolFromSmiles(text, sanitize=False)
    return mol is not None and mol.GetNumAtoms() > 0


def pubchem_search(query: str) -> dict:
    """
    Search PubChem for a compound name / keyword.
    Five-step cascade with error collection at every branch.
    Returns dict with keys: found, cid, name, smiles, iupac_name, mw, mf, source, error.
    """
    _BASE    = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    _PROPS   = "IUPACName,MolecularFormula,MolecularWeight,IsomericSMILES,CanonicalSMILES"
    _HDR     = {"User-Agent": "pKaNET-Cloud-Streamlit/1.0", "Accept": "application/json"}
    _TIMEOUT = 20
    errors   = []
    q_enc    = _url_quote(query, safe="")

    def _fetch(url):
        return _requests.get(url, headers=_HDR, timeout=_TIMEOUT)

    def _props_from_cid(cid):
        r = _fetch(f"{_BASE}/compound/cid/{cid}/property/{_PROPS}/JSON")
        r.raise_for_status()
        return r.json()["PropertyTable"]["Properties"][0]

    def _pack(props, source_label, matched_name):
        smi = (props.get("IsomericSMILES") or
               props.get("CanonicalSMILES") or
               next((v for k, v in props.items()
                     if "smiles" in k.lower() and v), None))
        if not smi:
            errors.append(f"_pack: no SMILES in props keys={list(props.keys())}")
            return None
        return dict(
            found=True, cid=props.get("CID"), name=matched_name,
            smiles=smi, iupac_name=props.get("IUPACName", ""),
            mw=props.get("MolecularWeight", ""),
            mf=props.get("MolecularFormula", ""),
            source=source_label, error=None,
        )

    # Step 1: name → CIDs → props by CID
    try:
        r = _fetch(f"{_BASE}/compound/name/{q_enc}/cids/JSON")
        if r.status_code == 200:
            body = r.json()
            if "IdentifierList" in body:
                cid    = body["IdentifierList"]["CID"][0]
                result = _pack(_props_from_cid(cid), "PubChem (name → CID)", query)
                if result: return result
            elif "Fault" in body:
                errors.append(f"Step1 Fault: {body['Fault'].get('Message','')}")
            else:
                errors.append(f"Step1 unexpected keys: {list(body.keys())}")
        else:
            errors.append(f"Step1 HTTP {r.status_code}: {r.text[:120]}")
    except Exception as e:
        errors.append(f"Step1 exception: {type(e).__name__}: {e}")

    # Step 2: name → property directly
    try:
        r = _fetch(f"{_BASE}/compound/name/{q_enc}/property/{_PROPS}/JSON")
        if r.status_code == 200:
            body = r.json()
            if "PropertyTable" in body:
                result = _pack(body["PropertyTable"]["Properties"][0],
                               "PubChem (name → property)", query)
                if result: return result
            elif "Fault" in body:
                errors.append(f"Step2 Fault: {body['Fault'].get('Message','')}")
            else:
                errors.append(f"Step2 unexpected keys: {list(body.keys())}")
        else:
            errors.append(f"Step2 HTTP {r.status_code}: {r.text[:80]}")
    except Exception as e:
        errors.append(f"Step2 exception: {type(e).__name__}: {e}")

    # Step 3: autocomplete → top suggestion
    try:
        r = _fetch(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/"
            f"compound/{q_enc}/JSON?limit=5"
        )
        if r.status_code == 200:
            suggestions = r.json().get("dictionary_terms", {}).get("compound", [])
            if not suggestions:
                errors.append("Step3: autocomplete returned 0 suggestions")
            for suggestion in suggestions:
                try:
                    r2 = _fetch(
                        f"{_BASE}/compound/name/"
                        f"{_url_quote(suggestion, safe='')}/cids/JSON"
                    )
                    if r2.status_code == 200:
                        body2 = r2.json()
                        if "IdentifierList" in body2:
                            cid    = body2["IdentifierList"]["CID"][0]
                            result = _pack(
                                _props_from_cid(cid),
                                f"PubChem (autocomplete → '{suggestion}')",
                                suggestion,
                            )
                            if result: return result
                except Exception as e2:
                    errors.append(f"Step3 suggestion '{suggestion}': {e2}")
                    continue
        else:
            errors.append(f"Step3 HTTP {r.status_code}: {r.text[:80]}")
    except Exception as e:
        errors.append(f"Step3 exception: {type(e).__name__}: {e}")

    # Step 4: word search (fuzzy)
    try:
        r = _fetch(f"{_BASE}/compound/name/{q_enc}/property/{_PROPS}/JSON?name_type=word")
        if r.status_code == 200:
            body = r.json()
            if "PropertyTable" in body:
                result = _pack(body["PropertyTable"]["Properties"][0],
                               "PubChem (word search)", query)
                if result: return result
            elif "Fault" in body:
                errors.append(f"Step4 Fault: {body['Fault'].get('Message','')}")
            else:
                errors.append(f"Step4 unexpected keys: {list(body.keys())}")
        else:
            errors.append(f"Step4 HTTP {r.status_code}: {r.text[:80]}")
    except Exception as e:
        errors.append(f"Step4 exception: {type(e).__name__}: {e}")

    # Step 5: full compound JSON fallback
    try:
        r = _fetch(f"{_BASE}/compound/name/{q_enc}/JSON")
        if r.status_code == 200:
            body = r.json()
            pc   = body.get("PC_Compounds", [{}])[0]
            cid  = pc.get("id", {}).get("id", {}).get("cid")
            smi  = None
            for prop in pc.get("props", []):
                if prop.get("urn", {}).get("label") == "SMILES":
                    smi = prop.get("value", {}).get("sval") or smi
            if smi and cid:
                return dict(found=True, cid=cid, name=query, smiles=smi,
                            iupac_name="", mw="", mf="",
                            source="PubChem (full compound JSON)", error=None)
            else:
                errors.append(f"Step5: CID={cid}, SMILES found={smi is not None}")
        else:
            errors.append(f"Step5 HTTP {r.status_code}")
    except Exception as e:
        errors.append(f"Step5 exception: {type(e).__name__}: {e}")

    err_detail = " | ".join(errors) if errors else "No error details (all silent)"
    return dict(
        found=False, cid=None, name=query, smiles="",
        iupac_name="", mw="", mf="", source=None,
        error=f"Not found on PubChem: '{query}' — {err_detail}",
    )


@st.cache_data(show_spinner=False, ttl=3600)
def cached_pubchem_search(query: str) -> dict:
    """Wrapper with Streamlit cache to avoid repeat lookups on every rerun."""
    return pubchem_search(query)


def resolve_text_input(raw: str, fallback_name: str = "ligand") -> tuple:
    """
    Resolve a text input to (smiles, name, pubchem_info_or_None).
    Accepts either a SMILES string (optionally `SMILES name`) or a compound name.
    Returns (None, None, None) on empty input.
    """
    raw = (raw or "").strip()
    if not raw:
        return None, None, None
    parts = raw.split(None, 1)
    token = parts[0]
    rest  = parts[1].strip() if len(parts) > 1 else ""
    if _is_smiles(token):
        name = _safe_name(rest) if rest else fallback_name
        return token, name, None
    # Treat as compound name → PubChem lookup
    pc = cached_pubchem_search(raw)
    if not pc["found"]:
        return None, None, pc
    name = _safe_name(pc["name"]) or fallback_name
    return pc["smiles"], name, pc


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
input_type = st.sidebar.selectbox(
    "Input type",
    ["Paste SMILES", "Search PubChem", "Upload SMI file", "Upload 3D structure", "Convert to SMILES"],
    help=(
        "Paste SMILES — type/paste a SMILES string directly.\n"
        "Search PubChem — type a compound name; resolved via PubChem.\n"
        "Upload SMI file — a .smi/.txt file, one `SMILES [name]` per line.\n"
        "Upload 3D structure — a .pdb / .mol2 / .sdf file.\n"
        "Convert to SMILES — paste names / InChI / SMILES (one per line), or an "
        "image/PDF; each line is auto-routed to the right converter.\n"
        "Upload PDF / Image — a PDF page or a photo/screenshot showing a "
        "scaffold + R-group table; build SMILES from it, then run the same "
        "analysis as any other input."
    ),
)
target_pH   = st.sidebar.slider("Target pH", 2.0, 12.0, 7.4, 0.1)
output_name = st.sidebar.text_input(
    "Output name", value="ligand",
    help="Base name for single-molecule inputs. Ignored for multi-molecule "
         "batches (SMI file / PDF/Image), which use each row's own Compound ID.",
)

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

# These are populated when the user enters a compound name resolved via PubChem.
resolved_smiles_from_name = None
resolved_pubchem_info     = None
resolved_name_for_output  = None

# Populated only by the "Upload PDF / Image" mode — a ready-to-run .smi-format byte
# string (built from the validated compound table) that feeds run_job()
# exactly like a hand-uploaded SMI file would.
pdf2smi_smi_bytes = None
paste_smi_bytes = None      # multi-line "Paste SMILES" -> routed through the SMI_FILE path
paste_smi_records = None    # parsed (smiles, name) records, for the overview grid
struct2smi_records = None   # resolved (smiles, name) from struct2smi (name/InChI/image)
struct2smi_bytes = None

if input_type == "Paste SMILES":
    text_in = st.text_area(
        "SMILES   (one per line for a batch)   example: `CC(=O)OC1=CC=CC=C1C(=O)O`",
        value="CC(=O)OC1=CC=CC=C1C(=O)O",
        height=120,
        help="Paste one SMILES per line. Each line may be `SMILES name` to set a custom "
             "name. A single line runs one molecule; multiple lines run as a batch.",
    )
    if text_in and text_in.strip():
        recs = [(smi, nm) for smi, nm in parse_smi_lines(text_in) if _is_smiles(smi)]
        n_bad = len([ln for ln in text_in.splitlines()
                     if ln.strip() and not ln.strip().startswith("#")]) - len(recs)
        if not recs:
            st.error("❌ No valid SMILES found in the box.")
            st.info("💡 Looking for a compound by name instead? Switch **Input type** to "
                    "**Search PubChem**.")
        elif len(recs) == 1:
            smiles_text = text_in.strip()            # single molecule -> existing path
            if n_bad:
                st.warning(f"⚠️ Ignored {n_bad} line(s) that weren't valid SMILES.")
        else:
            smiles_text = text_in.strip()            # truthy -> passes the run guard
            paste_smi_records = recs
            paste_smi_bytes = ("\n".join(f"{smi}\t{nm}" for smi, nm in recs) + "\n").encode("utf-8")
            msg = f"✅ Parsed {len(recs)} SMILES — will run as a batch."
            if n_bad:
                msg += f"  (ignored {n_bad} invalid line(s))"
            st.caption(msg)

elif input_type == "Search PubChem":
    name_in = st.text_input(
        "Compound name   examples: `aspirin`, `baicalein`, `quercetin`",
        value="aspirin",
        help="Looked up on PubChem (name → CID → properties, with autocomplete/fuzzy fallbacks).",
    )
    if name_in and name_in.strip():
        with st.spinner(f"🔍 Looking up '{name_in.strip()}' on PubChem…"):
            resolved_smiles_from_name, resolved_name_for_output, resolved_pubchem_info = \
                resolve_text_input(name_in, fallback_name=output_name or "ligand")
        if resolved_smiles_from_name:
            pc = resolved_pubchem_info
            st.success(f"✅ {pc['source']}")
            colA, colB = st.columns([2, 3])
            with colA:
                st.markdown(
                    f"- **CID:** [{pc['cid']}](https://pubchem.ncbi.nlm.nih.gov/compound/{pc['cid']})\n"
                    f"- **IUPAC:** {pc['iupac_name'] or '—'}\n"
                    f"- **Formula:** {pc['mf'] or '—'}\n"
                    f"- **MW:** {pc['mw'] or '—'}"
                )
                st.code(pc["smiles"], language="text")
            with colB:
                if DRAW_AVAILABLE:
                    mol_pc = Chem.MolFromSmiles(pc["smiles"])
                    if mol_pc:
                        AllChem.Compute2DCoords(mol_pc)
                        img_pc = Draw.MolToImage(mol_pc, size=(360, 260))
                        if img_pc:
                            st.image(img_pc, caption="2D preview (from PubChem SMILES)")
            smiles_text = pc["smiles"]
        else:
            err_msg = (
                resolved_pubchem_info.get("error")
                if resolved_pubchem_info
                else f"Not found on PubChem: '{name_in.strip()}'"
            )
            st.error(f"❌ PubChem lookup failed.\n\n{err_msg}")
            st.info(
                "💡 Possible fixes:\n"
                "- Check spelling (e.g. `baicalein`, `quercetin`, `aspirin`)\n"
                "- Switch **Input type** to **Paste SMILES** and paste the structure directly\n"
                "- Check your internet connection"
            )

elif input_type == "Upload SMI file":
    uploaded = st.file_uploader("Upload .smi file (SMILES [name] per line)", type=["smi", "txt"])
    st.info("📝 Format: `SMILES [optional_name]` per line")

elif input_type == "Upload 3D structure":
    uploaded = st.file_uploader("Upload ligand file", type=["pdb", "mol2", "sdf"])
    st.info("📝 PDB files are converted to canonical SMILES via Open Babel before processing.")

elif input_type == "Convert to SMILES":
    st.markdown("#### Convert names / InChI / SMILES (or an image) → SMILES")
    if not _STRUCT2SMI_OK:
        st.error("❌ struct2smi.py is not available on the server.")
    else:
        _bk = struct2smi.available_backends()
        st.caption(
            "Backends — name→structure (OPSIN): "
            + ("✅" if _bk["name_opsin"] else "❌ add py2opsin + a JRE")
            + "  ·  image OCSR: "
            + ("✅" if _bk["image_ocsr"] else "❌ stub (use the manual PDF/Image mode)")
        )
        conv_text = st.text_area(
            "One per line: chemical name, InChI, or SMILES (optionally `SMILES name`)",
            height=140,
            help="Each line is auto-routed: SMILES/InChI resolve locally; names need "
                 "OPSIN; images use OCSR (currently stubbed).",
        )
        conv_file = st.file_uploader(
            "…or an image / PDF (OCSR — experimental, currently stubbed)",
            type=["png", "jpg", "jpeg", "webp", "pdf"], key="struct2smi_file",
        )
        if (conv_text and conv_text.strip()) or conv_file is not None:
            _res = struct2smi.convert(
                text=conv_text if (conv_text and conv_text.strip()) else None,
                file_bytes=conv_file.getvalue() if conv_file is not None else None,
                filename=conv_file.name if conv_file is not None else None,
            )
            st.dataframe(
                pd.DataFrame([{"input": r.raw, "name": r.name, "kind": r.kind,
                               "status": r.status, "SMILES": r.smiles or "",
                               "note": r.note} for r in _res]),
                use_container_width=True, hide_index=True,
            )
            struct2smi_records = struct2smi.ok_records(_res)
            if struct2smi_records:
                struct2smi_bytes = struct2smi.records_to_smi_bytes(_res)
                st.success(f"✅ {len(struct2smi_records)} structure(s) resolved and ready.")
            else:
                st.warning("No structures resolved yet — names need OPSIN, images need "
                           "an OCSR backend.")

else:  # input_type == "Upload PDF / Image"
    st.markdown("#### 1 · Upload the PDF page or image with your scaffold + R-group table")
    pdf_file = st.file_uploader(
        "PDF or image file (PDF / PNG / JPG / WEBP / HEIC)",
        type=["pdf", "png", "jpg", "jpeg", "webp", "heic", "heif"],
        key="pdf2smi_uploader",
    )
    if pdf_file is not None:
        st.session_state["pdf2smi_bytes"]    = pdf_file.getvalue()
        st.session_state["pdf2smi_filename"] = pdf_file.name
    pdf_bytes_in_state = st.session_state.get("pdf2smi_bytes")
    filename_in_state  = st.session_state.get("pdf2smi_filename", "")
    is_pdf = filename_in_state.lower().endswith(".pdf")

    if not pdf_bytes_in_state:
        st.info("Upload a PDF or image to render it here, then describe the scaffold(s) below.")
    elif is_pdf and not _PYMUPDF_OK:
        st.error(
            "❌ PyMuPDF (`fitz`) is not installed on this server — PDF rendering "
            "is unavailable. Add `pymupdf` to `requirements.txt` and redeploy, "
            "or upload a PNG/JPG screenshot of the page instead."
        )
    else:
        if is_pdf:
            n_pages = pdf2smi_get_page_count(pdf_bytes_in_state)
            col_a, col_b = st.columns([1, 3])
            with col_a:
                page_num = st.number_input("Page", min_value=1, max_value=max(n_pages, 1),
                                            value=1, step=1, key="pdf2smi_page")
                dpi = st.slider("Render DPI", 100, 400, 200, step=50, key="pdf2smi_dpi")
                st.caption(f"{n_pages} page(s) in this PDF.")
            with col_b:
                img = pdf2smi_render_page(pdf_bytes_in_state, int(page_num), dpi)
                st.image(img, use_container_width=True)
            with st.expander("Extracted text layer (cross-check R-group names / IDs)"):
                pg_text = pdf2smi_extract_text(pdf_bytes_in_state, int(page_num))
                if pg_text:
                    st.code(pg_text, language=None)
                else:
                    st.info(
                        "No extractable text on this page — it's likely a flattened/"
                        "outlined export. Read the compound IDs and R-group names "
                        "visually from the image above instead."
                    )
        else:
            try:
                img = Image.open(io.BytesIO(pdf_bytes_in_state))
                st.image(img, use_container_width=True)
                st.caption(f"{filename_in_state} · {img.size[0]}×{img.size[1]} px")
            except Exception as e:
                magic = pdf_bytes_in_state[:16]
                looks_like = (
                    "HEIC/HEIF (iPhone photo format)" if b"ftyp" in magic and
                        (b"heic" in magic or b"heix" in magic or b"mif1" in magic) else
                    "AVIF" if b"ftypavif" in magic else
                    f"an unrecognized format (first bytes: {magic!r})"
                )
                st.error(f"❌ Could not open image: {e}")
                if "HEIC" in looks_like and not _HEIF_OK:
                    st.warning(
                        "This looks like a **HEIC/HEIF** photo (the default format for "
                        "iPhone camera shots), and the `pillow-heif` plugin needed to "
                        "decode it isn't installed on this server. Add `pillow-heif` to "
                        "`requirements.txt` and redeploy — or for now, re-export/share the "
                        "photo as PNG/JPEG (e.g. iPhone Settings → Camera → Formats → "
                        "'Most Compatible') and re-upload."
                    )
                else:
                    st.warning(
                        f"This appears to be {looks_like}, which isn't a format this "
                        "server's Pillow install can decode. Try converting it to PNG "
                        "or JPEG and re-uploading."
                    )
            st.info(
                "📝 No text layer to cross-check on an image upload — read the "
                "compound IDs and R-group names directly from the picture above."
            )

        if "pdf2smi_table_version" not in st.session_state:
            st.session_state["pdf2smi_table_version"] = 0
        _v = st.session_state["pdf2smi_table_version"]

        st.markdown("#### 2 · Describe the scaffold(s)")
        st.caption(
            "Dummy attachment atoms `[*:1]`, `[*:2]`, `[*:3]`, ... mark where "
            "each R-group attaches. **Slot_Roles** maps each slot, in order, "
            "to a column name in the Compounds table below — leave it blank "
            "for the default `R1,AR,R2`. **Repeat a role to attach the same "
            "fragment at more than one position** (e.g. `R1,R2,R1,R2` for a "
            "symmetric scaffold where both arms carry the same R1/R2 pair)."
        )
        if "pdf2smi_templates_df" not in st.session_state:
            st.session_state["pdf2smi_templates_df"] = pd.DataFrame(
                columns=["Template", "Scaffold_SMILES", "Slot_Roles"])
        st.session_state["pdf2smi_templates_df"] = st.data_editor(
            st.session_state["pdf2smi_templates_df"], num_rows="dynamic",
            use_container_width=True, key=f"pdf2smi_templates_editor_{_v}",
            column_config={
                "Template": st.column_config.TextColumn(required=True),
                "Scaffold_SMILES": st.column_config.TextColumn(required=True, width="large"),
                "Slot_Roles": st.column_config.TextColumn(
                    required=False,
                    help="Comma-separated column names, one per [*:n] slot in "
                         "order. Blank = default R1,AR,R2. Repeat a name to "
                         "reuse the same fragment at multiple slots, e.g. "
                         "R1,R2,R1,R2.",
                ),
            },
        )

        st.markdown("#### 3 · Fragment library (named, reusable R-groups)")
        st.caption(
            "Each SMILES is written **attachment-atom first** "
            "(e.g. para-hydroxyphenyl = `c1ccc(O)cc1`, not `Oc1ccccc1`)."
        )
        if "pdf2smi_library_df" not in st.session_state:
            st.session_state["pdf2smi_library_df"] = pd.DataFrame(columns=["Name", "SMILES"])
        st.session_state["pdf2smi_library_df"] = st.data_editor(
            st.session_state["pdf2smi_library_df"], num_rows="dynamic",
            use_container_width=True, key=f"pdf2smi_library_editor_{_v}",
            column_config={
                "Name": st.column_config.TextColumn(required=True),
                "SMILES": st.column_config.TextColumn(required=True),
            },
        )

        st.markdown("#### 4 · Compounds")
        st.caption(
            "R1 / R2 / AR cells accept a name from the fragment library above, "
            "*or* a raw SMILES typed directly. Leave R2 blank for scaffolds "
            "that don't use it."
        )
        if "pdf2smi_compounds_df" not in st.session_state:
            st.session_state["pdf2smi_compounds_df"] = pd.DataFrame(
                columns=["Compound_ID", "Template", "R1", "R2", "AR"])
        st.session_state["pdf2smi_compounds_df"] = st.data_editor(
            st.session_state["pdf2smi_compounds_df"], num_rows="dynamic",
            use_container_width=True, key=f"pdf2smi_compounds_editor_{_v}",
            column_config={
                "Compound_ID": st.column_config.TextColumn(required=True),
                "Template": st.column_config.TextColumn(required=True),
                "R1": st.column_config.TextColumn(),
                "R2": st.column_config.TextColumn(),
                "AR": st.column_config.TextColumn(),
            },
        )

        load_ex_col, load_ex2_col, clear_col, build_col = st.columns(4)
        if load_ex_col.button("📋 Load example (lignin imidazoles)", use_container_width=True):
            st.session_state["pdf2smi_templates_df"] = pd.DataFrame([
                {"Template": "core_2",    "Scaffold_SMILES": "Cc1nc([*:1])[nH]c1[*:2]",        "Slot_Roles": ""},
                {"Template": "core_2_ND", "Scaffold_SMILES": "Cc1nc([*:1])n([2H])c1[*:2]",      "Slot_Roles": ""},
                {"Template": "core_3",    "Scaffold_SMILES": "Cc1nc([*:1])n([*:3])c1[*:2]",     "Slot_Roles": ""},
            ])
            st.session_state["pdf2smi_library_df"] = pd.DataFrame([
                {"Name": "Ph",              "SMILES": "c1ccccc1"},
                {"Name": "4-NMe2Ph",        "SMILES": "c1ccc(N(C)C)cc1"},
                {"Name": "4-OHPh",          "SMILES": "c1ccc(O)cc1"},
                {"Name": "2-OHPh",          "SMILES": "c1ccccc1O"},
                {"Name": "4-ClPh",          "SMILES": "c1ccc(Cl)cc1"},
                {"Name": "4-NO2Ph",         "SMILES": "c1ccc([N+](=O)[O-])cc1"},
                {"Name": "2-NO2Ph",         "SMILES": "c1ccccc1[N+](=O)[O-]"},
                {"Name": "4-Py",            "SMILES": "c1ccncc1"},
                {"Name": "2-Py",            "SMILES": "c1ccccn1"},
                {"Name": "2-thienyl",       "SMILES": "c1cccs1"},
                {"Name": "2-furyl",         "SMILES": "c1ccco1"},
                {"Name": "guaiacyl",        "SMILES": "c1ccc(O)c(OC)c1"},
                {"Name": "syringyl",        "SMILES": "c1cc(OC)c(O)c(OC)c1"},
                {"Name": "1-pyrenyl",       "SMILES": "c1ccc2ccc3cccc4ccc1c2c34"},
                {"Name": "cyclohexyl",      "SMILES": "C1CCCCC1"},
                {"Name": "1-heptyl",        "SMILES": "CCCCCCC"},
                {"Name": "4-MeOPh",         "SMILES": "c1ccc(OC)cc1"},
                {"Name": "4-BrPh",          "SMILES": "c1ccc(Br)cc1"},
                {"Name": "2-MeO-3-OH-4-MeOPh", "SMILES": "c1c(OC)c(O)c(OC)cc1"},
                {"Name": "3,4-diFPh",       "SMILES": "c1ccc(F)c(F)c1"},
                {"Name": "3,4-diOHPh",      "SMILES": "c1ccc(O)c(O)c1"},
                {"Name": "5-CH2OH-furyl",   "SMILES": "c1ccc(CO)o1"},
                {"Name": "2-MeO-3-OH-4-OHPh", "SMILES": "c1c(OC)c(O)c(O)cc1"},
                {"Name": "pyrrol-2-yl",     "SMILES": "c1ccc[nH]1"},
                {"Name": "indol-3-yl",      "SMILES": "c1c[nH]c2ccccc12"},
            ])
            st.session_state["pdf2smi_compounds_df"] = pd.DataFrame([
                {"Compound_ID": "2a",   "Template": "core_2",    "R1": "Ph",                    "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2a-D", "Template": "core_2_ND", "R1": "Ph",                    "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2b",   "Template": "core_2",    "R1": "4-NMe2Ph",              "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2c",   "Template": "core_2",    "R1": "4-OHPh",                "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2d",   "Template": "core_2",    "R1": "2-OHPh",                "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2e",   "Template": "core_2",    "R1": "4-ClPh",                "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2f",   "Template": "core_2",    "R1": "4-NO2Ph",               "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2g",   "Template": "core_2",    "R1": "2-NO2Ph",               "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2h",   "Template": "core_2",    "R1": "4-Py",                  "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2i",   "Template": "core_2",    "R1": "2-Py",                  "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2j",   "Template": "core_2",    "R1": "2-thienyl",             "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2k",   "Template": "core_2",    "R1": "2-furyl",               "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2l",   "Template": "core_2",    "R1": "guaiacyl",              "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2m",   "Template": "core_2",    "R1": "1-pyrenyl",             "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2n",   "Template": "core_2",    "R1": "cyclohexyl",            "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2o",   "Template": "core_2",    "R1": "1-heptyl",              "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2p",   "Template": "core_2",    "R1": "2-MeO-3-OH-4-MeOPh",    "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2q",   "Template": "core_2",    "R1": "3,4-diFPh",             "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2r",   "Template": "core_2",    "R1": "3,4-diOHPh",            "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2s",   "Template": "core_2",    "R1": "5-CH2OH-furyl",         "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2t",   "Template": "core_2",    "R1": "2-MeO-3-OH-4-OHPh",     "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2u",   "Template": "core_2",    "R1": "pyrrol-2-yl",           "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "2w",   "Template": "core_2",    "R1": "indol-3-yl",            "R2": "",      "AR": "syringyl"},
                {"Compound_ID": "3a",   "Template": "core_3",    "R1": "Ph",                    "R2": "4-BrPh","AR": "syringyl"},
                {"Compound_ID": "3b",   "Template": "core_3",    "R1": "4-MeOPh",               "R2": "Ph",    "AR": "syringyl"},
                {"Compound_ID": "4a",   "Template": "core_2",    "R1": "Ph",                    "R2": "",      "AR": "guaiacyl"},
            ])
            st.session_state["pdf2smi_table_version"] = _v + 1
            st.rerun()

        if load_ex2_col.button("📋 Load example (symmetric R-groups)", use_container_width=True):
            # Catechol bis-ether where BOTH arms carry the same NR1R2 amide --
            # demonstrates Slot_Roles repetition: "R1,R2,R1,R2" attaches the
            # same R1 fragment to slots 1 & 3, and the same R2 to slots 2 & 4.
            st.session_state["pdf2smi_templates_df"] = pd.DataFrame([
                {"Template": "catechol_bisamide",
                 "Scaffold_SMILES": "c1(OCC(=O)N([*:1])[*:2])c(OCC(=O)N([*:3])[*:4])cccc1",
                 "Slot_Roles": "R1,R2,R1,R2"},
            ])
            st.session_state["pdf2smi_library_df"] = pd.DataFrame([
                {"Name": "Me",       "SMILES": "C"},
                {"Name": "Et",       "SMILES": "CC"},
                {"Name": "n-Bu",     "SMILES": "CCCC"},
                {"Name": "n-C10H21", "SMILES": "CCCCCCCCCC"},
                {"Name": "i-Bu",     "SMILES": "CC(C)C"},
                {"Name": "Cy",       "SMILES": "C1CCCCC1"},
                {"Name": "Ph",       "SMILES": "c1ccccc1"},
                {"Name": "Bn",       "SMILES": "Cc1ccccc1"},
            ])
            st.session_state["pdf2smi_compounds_df"] = pd.DataFrame([
                {"Compound_ID": "1a", "Template": "catechol_bisamide", "R1": "Me",   "R2": "Me",   "AR": ""},
                {"Compound_ID": "1b", "Template": "catechol_bisamide", "R1": "Et",   "R2": "Et",   "AR": ""},
                {"Compound_ID": "1c", "Template": "catechol_bisamide", "R1": "n-Bu", "R2": "n-Bu", "AR": ""},
                {"Compound_ID": "1d", "Template": "catechol_bisamide", "R1": "n-C10H21", "R2": "n-C10H21", "AR": ""},
                {"Compound_ID": "1e", "Template": "catechol_bisamide", "R1": "i-Bu", "R2": "i-Bu", "AR": ""},
                {"Compound_ID": "1f", "Template": "catechol_bisamide", "R1": "Cy",   "R2": "Cy",   "AR": ""},
                {"Compound_ID": "1g", "Template": "catechol_bisamide", "R1": "Ph",   "R2": "Bn",   "AR": ""},
                {"Compound_ID": "1h", "Template": "catechol_bisamide", "R1": "Et",   "R2": "Cy",   "AR": ""},
            ])
            st.session_state["pdf2smi_table_version"] = _v + 1
            st.rerun()

        if clear_col.button("🗑️ Clear tables", use_container_width=True):
            st.session_state["pdf2smi_templates_df"] = pd.DataFrame(
                columns=["Template", "Scaffold_SMILES", "Slot_Roles"])
            st.session_state["pdf2smi_library_df"] = pd.DataFrame(columns=["Name", "SMILES"])
            st.session_state["pdf2smi_compounds_df"] = pd.DataFrame(
                columns=["Compound_ID", "Template", "R1", "R2", "AR"])
            st.session_state["pdf2smi_table_version"] = _v + 1
            st.rerun()

        if build_col.button("✨ Build & validate", type="primary", use_container_width=True):
            templates = dict(zip(st.session_state["pdf2smi_templates_df"]["Template"],
                                  st.session_state["pdf2smi_templates_df"]["Scaffold_SMILES"]))
            library = dict(zip(st.session_state["pdf2smi_library_df"]["Name"],
                                st.session_state["pdf2smi_library_df"]["SMILES"]))
            template_roles = {}
            if "Slot_Roles" in st.session_state["pdf2smi_templates_df"].columns:
                template_roles = dict(zip(st.session_state["pdf2smi_templates_df"]["Template"],
                                           st.session_state["pdf2smi_templates_df"]["Slot_Roles"]))
            result_df, mols, legends = pdf2smi_build_all(
                st.session_state["pdf2smi_compounds_df"], templates, library,
                template_roles=template_roles)
            st.session_state["pdf2smi_result_df"]      = result_df
            st.session_state["pdf2smi_result_mols"]     = mols
            st.session_state["pdf2smi_result_legends"]  = legends

        if "pdf2smi_result_df" in st.session_state:
            result_df = st.session_state["pdf2smi_result_df"]
            n_ok   = int((result_df["Status"] == "OK").sum()) if len(result_df) else 0
            n_fail = len(result_df) - n_ok

            m1, m2, m3 = st.columns(3)
            m1.metric("Compounds", len(result_df))
            m2.metric("Validated OK", n_ok)
            m3.metric("Failed", n_fail)

            st.dataframe(result_df, use_container_width=True, hide_index=True)
            if n_fail:
                st.warning("Some rows failed to build — check the Error column above.")

            if st.session_state.get("pdf2smi_result_mols"):
                with st.expander("Structure grid preview", expanded=True):
                    grid = pdf2smi_make_grid_image(
                        st.session_state["pdf2smi_result_mols"],
                        st.session_state["pdf2smi_result_legends"],
                    )
                    if grid:
                        st.image(grid)

            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "⬇️ Download .smi", data=pdf2smi_to_smi_bytes(result_df),
                file_name="pdf_extracted.smi", mime="text/plain", use_container_width=True,
            )
            dl2.download_button(
                "⬇️ Download .csv", data=pdf2smi_to_csv_bytes(result_df),
                file_name="pdf_extracted.csv", mime="text/csv", use_container_width=True,
            )

            if n_ok:
                pdf2smi_smi_bytes = pdf2smi_to_smi_bytes(result_df)
                st.success(
                    f"✅ {n_ok} validated compound(s) ready — click "
                    "**🚀 Run Analysis** below to send them through the full pKaNET pipeline."
                )
            else:
                st.warning("⚠️ No validated compounds yet — fix the errors above and rebuild.")

# Live PDB preview
converted_smiles_from_pdb = None
if input_type == "Upload 3D structure" and uploaded is not None and uploaded.name.endswith(".pdb"):
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
        default_show = 5
        n_total = len(top_microstates)
        with st.expander(f"📊 Ranked microstate table ({n_total} states)", expanded=True):
            if n_total > default_show:
                show_all = st.checkbox(
                    f"Show all {n_total} microstates",
                    value=False,
                    key=f"show_all_micro_{idx}",
                )
                display_list = top_microstates if show_all else top_microstates[:default_show]
                if not show_all:
                    st.caption(f"Showing top {default_show} of {n_total} — tick the box above to see all.")
            else:
                display_list = top_microstates
            show_microstate_table(display_list)
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

# ---- Combined 2D structure overview (R/S annotated) for list inputs ----
_grid_records = None
if input_type == "Upload SMI file" and uploaded is not None:
    try:
        _grid_records = parse_smi_lines(uploaded.getvalue().decode("utf-8", errors="replace"))
    except Exception:
        _grid_records = None
elif input_type == "Paste SMILES" and paste_smi_records and len(paste_smi_records) > 1:
    _grid_records = paste_smi_records
elif input_type == "Convert to SMILES" and struct2smi_records:
    _grid_records = struct2smi_records

if _grid_records:
    _n_grid = len(_grid_records)
    with st.expander(f"🧪 Structure overview — {_n_grid} molecule(s), R/S annotated", expanded=True):
        _grid_img = make_structure_grid_image(_grid_records, max_mols=60)
        if _grid_img is not None:
            st.image(_grid_img, use_container_width=True)
            if _n_grid > 60:
                st.caption(f"Showing the first 60 of {_n_grid} structures.")
        else:
            st.info("No valid structures to display.")

run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

if run_btn:
    if   input_type == "Paste SMILES"      and not smiles_text:
        st.error("⚠️ Please enter a SMILES string")
    elif input_type == "Search PubChem"    and not smiles_text:
        st.error("⚠️ Please enter a compound name that PubChem can resolve")
    elif input_type == "Upload SMI file"   and not uploaded:
        st.error("⚠️ Please upload a .smi file")
    elif input_type == "Upload 3D structure" and not uploaded:
        st.error("⚠️ Please upload a ligand file")
    elif input_type == "Upload 3D structure" and uploaded.name.endswith(".pdb") and converted_smiles_from_pdb is None:
        st.error("⚠️ PDB → SMILES conversion failed. Cannot proceed.")
    elif input_type == "Upload PDF / Image" and not pdf2smi_smi_bytes:
        st.error("⚠️ Upload a PDF and click **✨ Build & validate** to get at least one "
                  "validated compound before running.")
    elif input_type == "Convert to SMILES" and not struct2smi_bytes:
        st.error("⚠️ Provide at least one resolvable name / InChI / SMILES "
                  "(or a supported image).")
    elif not output_formats:
        st.error("⚠️ Please select at least one output format")
    else:
        try:
            with st.spinner("🔬 Enumerating tautomers, scoring microstates, building 3D structures…"):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)

                    if input_type == "Upload 3D structure" and converted_smiles_from_pdb:
                        eff_type   = "SMILES"
                        eff_smiles = converted_smiles_from_pdb
                        eff_bytes  = None
                        eff_name   = None
                        st.info(f"🔄 Using SMILES from PDB: `{converted_smiles_from_pdb}`")
                    elif input_type == "Upload 3D structure":
                        eff_type   = "FILE"
                        eff_smiles = None
                        eff_bytes  = uploaded.read()
                        eff_name   = uploaded.name
                    elif input_type == "Paste SMILES" and paste_smi_bytes:
                        eff_type   = "SMI_FILE"
                        eff_smiles = None
                        eff_bytes  = paste_smi_bytes
                        eff_name   = "pasted.smi"
                        st.info(f"🔄 Running {len(paste_smi_records)} pasted SMILES as a batch.")
                    elif input_type in ("Paste SMILES", "Search PubChem"):
                        # smiles_text is either a raw SMILES string (Paste SMILES) or a
                        # PubChem-resolved SMILES (Search PubChem).
                        eff_type   = "SMILES"
                        eff_smiles = smiles_text
                        eff_bytes  = None
                        eff_name   = None
                        if resolved_pubchem_info and resolved_pubchem_info.get("found"):
                            st.info(
                                f"🔄 Using PubChem-resolved SMILES "
                                f"(CID {resolved_pubchem_info['cid']}): `{smiles_text}`"
                            )
                    elif input_type == "Convert to SMILES":
                        eff_type   = "SMI_FILE"
                        eff_smiles = None
                        eff_bytes  = struct2smi_bytes
                        eff_name   = "struct2smi.smi"
                        st.info(f"🔄 Running {len(struct2smi_records)} converted structure(s).")
                    elif input_type == "Upload PDF / Image":
                        eff_type   = "SMI_FILE"
                        eff_smiles = None
                        eff_bytes  = pdf2smi_smi_bytes
                        eff_name   = "pdf_extracted.smi"
                        st.info("🔄 Using SMILES built from the uploaded PDF.")
                    else:  # "Upload SMI file"
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

                    with st.expander("📊 Summary", expanded=False):
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
                        zip_minimized_structures(str(out_dir_path), str(zip_sel), output_formats, rank_only=1)
                        label = (f"🧬 Download {output_formats[0]} files \u2013 Rank 1 (ZIP)"
                                 if len(output_formats) == 1
                                 else f"🧬 Download {' + '.join(output_formats)} \u2013 Rank 1 (ZIP)")
                        st.download_button(
                            label,
                            data=zip_sel.read_bytes(),
                            file_name=f"pkanet_rank1_{'_'.join(f.lower() for f in output_formats)}.zip",
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
**By compound name** (PubChem auto-lookup):
```
aspirin
baicalein
quercetin
```
**Acylhydrazone** (amide NH preserved, charge 0):
```
O=C(N/N=C/CC)C1=NC(C(N/N=C/CC)=O)=CC=C1
```
**Glycine** (zwitterion at pH 7.4):
```
NCC(=O)O
```

""")

st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#666;font-size:0.9rem;'>"
    f"<img src='{_HEADER_LOGO_URL}' style='width:20px;height:20px;vertical-align:middle;margin-right:5px;' />"
    f"pKaNET Cloud+ — "
    f"<a href='mailto:kowith@ccs.tsukuba.ac.jp'>kowith@ccs.tsukuba.ac.jp</a>"
    f"</div>",
    unsafe_allow_html=True,
)
