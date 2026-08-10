"""
struct2smi.py — front-end converter that turns the various ways a structure is
represented (in papers or user input) into NORMALIZED SMILES records, which then
feed the existing pKaNET pipeline (app.py / core.py) through the SMI_FILE path.

Architecture:
    INPUT -> [1] ROUTER (classify) -> [2] HANDLER (per kind) -> [3] NORMALIZE -> records

Design goals (deliberately LIGHTWEIGHT -- deployable as-is):
  * SMILES / InChI       -> handled by RDKit alone (always available)
  * chemical name/IUPAC  -> handled by OPSIN via py2opsin  (OPTIONAL dep; needs a JRE)
  * image / PDF (OCSR)   -> STUB. Plug DECIMER/MolScribe into _ocsr_backend() later.
Nothing here imports torch or any heavy ML, so core.py stays free of these deps.
Every handler DEGRADES GRACEFULLY: if a backend is missing it returns a
`needs_review` record (never a silently-wrong structure), so app.py can fall back
to the manual entry path.

Public API (what app.py calls):
    convert(text=..., file_bytes=..., filename=...) -> list[StructResult]
    ok_records(results)          -> list[(smiles, name)]
    records_to_smi_bytes(results)-> bytes   (SMILES<TAB>name per line, for run_job)
    available_backends()         -> dict[str, bool]
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")   # silence expected parse-failure spam during routing

# -- optional OPSIN (name -> structure) --------------------------------------
try:
    from py2opsin import py2opsin as _opsin
    _OPSIN_IMPORTED = True
except Exception:
    _opsin = None
    _OPSIN_IMPORTED = False

# -- kinds -------------------------------------------------------------------
KIND_SMILES  = "smiles"
KIND_INCHI   = "inchi"
KIND_NAME    = "name"
KIND_IMAGE   = "image"
KIND_UNKNOWN = "unknown"

_INCHI_RE = re.compile(r"^InChI=1S?/", re.IGNORECASE)


@dataclass
class StructResult:
    name: str
    kind: str
    status: str                       # "ok" | "needs_review" | "failed"
    smiles: Optional[str] = None
    raw: str = ""                     # original input token / line
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# -- [3] NORMALIZE -----------------------------------------------------------
def _normalize(smiles: Optional[str], strip_salt: bool = True,
               assign_stereo: bool = True) -> Optional[str]:
    """RDKit sanitize + canonicalize (+ optional largest-fragment salt strip).
    Returns canonical SMILES or None. Every path out of a handler runs through
    this so downstream always gets a valid, canonical string."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        if strip_salt:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if len(frags) > 1:                      # drop counterions / solvent
                mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        if assign_stereo:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


# -- OPSIN availability (import AND a working Java runtime) -------------------
_opsin_checked: Optional[bool] = None

def _opsin_works() -> bool:
    """True only if py2opsin imports AND actually resolves (i.e. a JRE is present).
    Cached after the first probe so we pay the check once."""
    global _opsin_checked
    if _opsin_checked is None:
        if not _OPSIN_IMPORTED:
            _opsin_checked = False
        else:
            try:
                _opsin_checked = bool(_opsin("methane", output_format="SMILES"))
            except Exception:
                _opsin_checked = False
    return _opsin_checked


# -- [1] ROUTER --------------------------------------------------------------
def classify_line(line: str):
    """Classify one text line. Returns (kind, token, custom_name).
    For structures, token = parts[0] and custom_name = the rest (like a .smi line);
    for names (which may contain spaces) token = the whole line, custom_name = None."""
    line = line.strip()
    parts = line.split()
    head = parts[0]
    if _INCHI_RE.match(head):
        return KIND_INCHI, head, (parts[1] if len(parts) > 1 else None)
    if Chem.MolFromSmiles(head) is not None:            # parses -> it's a structure
        return KIND_SMILES, head, (parts[1] if len(parts) > 1 else None)
    return KIND_NAME, line, None                        # otherwise: chemical name


def classify_input(*, text=None, file_bytes=None, filename=None) -> str:
    """Coarse top-level kind: a file -> image/PDF (OCSR); text is routed per-line
    in convert()."""
    if file_bytes is not None:
        return KIND_IMAGE
    return KIND_UNKNOWN


# -- [2] HANDLERS ------------------------------------------------------------
def handle_smiles(tok: str, name: str) -> StructResult:
    smi = _normalize(tok)
    if smi:
        return StructResult(name, KIND_SMILES, "ok", smiles=smi, raw=tok)
    return StructResult(name, KIND_SMILES, "failed", raw=tok,
                        note="RDKit could not parse as SMILES")


def handle_inchi(tok: str, name: str) -> StructResult:
    try:
        mol = Chem.MolFromInchi(tok)
    except Exception:
        mol = None
    if mol is None:
        return StructResult(name, KIND_INCHI, "failed", raw=tok,
                            note="RDKit could not parse InChI")
    return StructResult(name, KIND_INCHI, "ok",
                        smiles=_normalize(Chem.MolToSmiles(mol)), raw=tok)


def handle_name(tok: str, name: str) -> StructResult:
    if not _opsin_works():
        return StructResult(name, KIND_NAME, "needs_review", raw=tok,
                            note="name->structure needs OPSIN (add py2opsin + a Java "
                                 "runtime), or use the 'Search PubChem' mode")
    try:
        smi = _opsin(tok, output_format="SMILES")
    except Exception:
        smi = ""
    smi = _normalize(smi) if smi else None
    if smi:
        return StructResult(name, KIND_NAME, "ok", smiles=smi, raw=tok)
    return StructResult(name, KIND_NAME, "failed", raw=tok,
                        note="OPSIN could not resolve this name")


def handle_image(file_bytes: bytes, filename: str = "image") -> List[StructResult]:
    """STUB for OCSR (image/PDF -> SMILES). Returns needs_review so the app falls
    back to the manual/vision-assisted path rather than emitting wrong structures.
    Wire a real engine in _ocsr_backend() to activate this."""
    if _ocsr_available():
        try:
            out = []
            for i, smi in enumerate(_ocsr_backend(file_bytes, filename), start=1):
                smi = _normalize(smi)
                out.append(StructResult(f"{filename}_{i:03d}", KIND_IMAGE,
                                        "ok" if smi else "failed", smiles=smi,
                                        raw=filename))
            return out or [StructResult(filename, KIND_IMAGE, "failed", raw=filename,
                                        note="OCSR returned nothing")]
        except Exception as e:
            return [StructResult(filename, KIND_IMAGE, "failed", raw=filename,
                                 note=f"OCSR error: {e}")]
    return [StructResult(filename, KIND_IMAGE, "needs_review", raw=filename,
                         note="OCSR backend not installed. On a capable host run "
                              "`pip install -r requirements-ocsr.txt` (DECIMER) — it "
                              "won't run on Streamlit Cloud. Or use the manual "
                              "scaffold+R-group entry in the 'Upload PDF / Image' mode.")]


# -- OCSR (image/PDF -> SMILES) via DECIMER: optional & LAZILY imported -------
# DECIMER (TensorFlow-based) is HEAVY and downloads model weights on first use.
# It is deliberately NOT imported at module load: the module stays lightweight
# and deployable, and DECIMER is only touched when an image is actually handled.
# Install it separately (see requirements-ocsr.txt) on a capable host --
# it will NOT build/run on Streamlit Community Cloud.
import importlib.util as _ilu

_decimer_predict = None
_decimer_loaded = False
_decimer_error = None
_segmenter = None
_segmenter_loaded = False


def _ocsr_installed() -> bool:
    """Cheap check (NO heavy import): is DECIMER importable? Used for the UI
    status caption so we don't load TensorFlow just to render a page."""
    try:
        return _ilu.find_spec("DECIMER") is not None
    except Exception:
        return False


def _load_decimer():
    """Lazily import DECIMER.predict_SMILES (loads TensorFlow -- slow, first call
    only; downloads model weights on first prediction)."""
    global _decimer_predict, _decimer_loaded, _decimer_error
    if not _decimer_loaded:
        _decimer_loaded = True
        try:
            from DECIMER import predict_SMILES
            _decimer_predict = predict_SMILES
        except Exception as e:
            _decimer_predict = None
            _decimer_error = f"{type(e).__name__}: {e}"
    return _decimer_predict


def _load_segmenter():
    """Optional DECIMER-Segmentation: splits a multi-structure figure into single
    structures BEFORE OCSR. If absent, the whole page is treated as one structure
    (fine for an isolated single-molecule image, wrong for a grid/figure)."""
    global _segmenter, _segmenter_loaded
    if not _segmenter_loaded:
        _segmenter_loaded = True
        try:
            from decimer_segmentation import segment_chemical_structures
            _segmenter = segment_chemical_structures
        except Exception:
            _segmenter = None
    return _segmenter


def _ocsr_available() -> bool:
    return _ocsr_installed()


def _pdf_to_images(pdf_bytes: bytes, dpi: int = 300):
    """Rasterize PDF pages to numpy RGB arrays via PyMuPDF (already a dependency)."""
    import io as _io
    import pymupdf as fitz                        # (was `import fitz` — deprecated alias)
    import numpy as np
    from PIL import Image
    imgs = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            im = Image.open(_io.BytesIO(pix.tobytes("png"))).convert("RGB")
            imgs.append(np.array(im))
    finally:
        doc.close()
    return imgs


def _ocsr_backend(image_bytes: bytes, filename: str = "image"):
    """Real OCSR. image/PDF bytes -> list[SMILES].
    Pipeline: (PDF -> raster pages) -> (optional segmentation into single
    structures) -> DECIMER predict_SMILES per structure. Returns [] on total
    failure. NOTE: OCSR reads DRAWN single structures; it does NOT parse a
    scaffold + R-group *table* (Markush) -- that still needs the manual/enumerate
    path in the 'Upload PDF / Image' mode."""
    predict = _load_decimer()
    if predict is None:
        if _ocsr_installed():
            raise RuntimeError(
                "DECIMER is installed but failed to import — " + (_decimer_error or "unknown") +
                ". Most often a NumPy 2.x vs TensorFlow ABI clash: pin `numpy<2` "
                "in the requirements you deploy, then restart.")
        raise RuntimeError("DECIMER not installed (pip install -r requirements-ocsr.txt)")

    import io as _io
    import os
    import tempfile
    import numpy as np
    from PIL import Image

    is_pdf = (filename or "").lower().endswith(".pdf") or image_bytes[:5] == b"%PDF-"

    # 1) page image(s) as numpy RGB arrays
    if is_pdf:
        page_imgs = _pdf_to_images(image_bytes)
    else:
        page_imgs = [np.array(Image.open(_io.BytesIO(image_bytes)).convert("RGB"))]

    # 2) segment each page into single-structure crops (if segmenter installed)
    seg = _load_segmenter()
    structure_imgs = []
    for pim in page_imgs:
        if seg is not None:
            try:
                crops = seg(pim)                 # list of numpy arrays
                structure_imgs.extend(list(crops) if len(crops) else [pim])
            except Exception:
                structure_imgs.append(pim)
        else:
            structure_imgs.append(pim)           # no segmenter -> whole page = 1

    # 3) OCSR each structure image (DECIMER wants a file path)
    smiles_out = []
    for arr in structure_imgs:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            Image.fromarray(arr.astype("uint8")).save(tf.name)
            p = tf.name
        try:
            smi = predict(p)
            if smi:
                smiles_out.append(smi)
        except Exception:
            pass
        finally:
            try:
                os.unlink(p)
            except Exception:
                pass
    return smiles_out


# -- abbreviation dictionary (for future OCSR condensed-label expansion) -----
ABBREVIATIONS = {
    "Me": "C", "Et": "CC", "Pr": "CCC", "iPr": "C(C)C", "Bu": "CCCC",
    "tBu": "C(C)(C)C", "Ph": "c1ccccc1", "Bn": "Cc1ccccc1", "Ac": "C(C)=O",
    "Bz": "C(=O)c1ccccc1", "Ts": "S(=O)(=O)c1ccc(C)cc1", "Ms": "CS(=O)(=O)",
    "Boc": "C(=O)OC(C)(C)C", "Cbz": "C(=O)OCc1ccccc1", "TMS": "[Si](C)(C)C",
    "TFA": "C(=O)C(F)(F)F",
}


# -- public entry point ------------------------------------------------------
def convert(*, text: Optional[str] = None, file_bytes: Optional[bytes] = None,
            filename: Optional[str] = None, strip_salt: bool = True) -> List[StructResult]:
    """Main entry. Provide EITHER `text` (one representation per line -- chemical
    name / InChI / SMILES, a structure line may carry a trailing custom name like
    a .smi file) OR an uploaded `file_bytes` (image/PDF). Returns StructResults;
    feed the ok ones downstream via records_to_smi_bytes()."""
    results: List[StructResult] = []

    if file_bytes is not None:
        results.extend(handle_image(file_bytes, filename or "image"))
        return results

    if not text:
        return results

    idx = 1
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kind, token, custom = classify_line(line)
        name = custom or f"mol_{idx:03d}"
        if kind == KIND_INCHI:
            res = handle_inchi(token, name)
        elif kind == KIND_SMILES:
            res = handle_smiles(token, name)
        elif kind == KIND_NAME:
            res = handle_name(token, name)
        else:
            res = StructResult(name, KIND_UNKNOWN, "failed", raw=line,
                               note="could not classify")
        results.append(res)
        idx += 1
    return results


# -- helpers for app.py ------------------------------------------------------
def ok_records(results: List[StructResult]):
    """[(smiles, name), ...] for the resolved structures -- feeds the grid + run."""
    return [(r.smiles, r.name) for r in results if r.ok and r.smiles]


def records_to_smi_bytes(results: List[StructResult]) -> bytes:
    """Render resolved structures as .smi bytes (SMILES<TAB>name) -- the exact
    shape run_job() consumes for input_type='SMI_FILE'."""
    ok = ok_records(results)
    return ("\n".join(f"{s}\t{n}" for s, n in ok) + "\n").encode("utf-8") if ok else b""


def available_backends() -> dict:
    """Report which handlers are actually live (so the UI can show status)."""
    return {"smiles": True, "inchi": True,
            "name_opsin": _opsin_works(), "image_ocsr": _ocsr_available()}


# ── Markush enumeration (lightweight; RDKit only) ────────────────────────────
# Reads a scaffold with ONE attachment point + a table of R-groups, and builds
# one molecule per row. This is the deployable path for the "scaffold + R-group
# table" figures OCSR can't handle. No heavy deps.
RGROUP_SHORTHAND = {
    "Me": "C", "CH3": "C", "Et": "CC", "C2H5": "CC",
    "Pr": "CCC", "nPr": "CCC", "n-Pr": "CCC", "C3H7": "CCC", "iPr": "C(C)C", "i-Pr": "C(C)C",
    "Bu": "CCCC", "nBu": "CCCC", "C4H9": "CCCC", "tBu": "C(C)(C)C", "t-Bu": "C(C)(C)C",
    "OMe": "OC", "OCH3": "OC", "OEt": "OCC", "OBu": "OCCCC", "OnBu": "OCCCC",
    "OtBu": "OC(C)(C)C", "OC(CH3)3": "OC(C)(C)C", "OiPr": "OC(C)C",
    "F": "F", "Cl": "Cl", "Br": "Br", "I": "I",
    "CF3": "C(F)(F)F", "OCF3": "OC(F)(F)F",
    "NO2": "[N+](=O)[O-]", "CN": "C#N", "OH": "O", "NH2": "N", "SH": "S",
    "COOH": "C(=O)O", "CO2H": "C(=O)O", "COOEt": "C(=O)OCC", "CHO": "C=O",
    "Ac": "C(C)=O", "COCH3": "C(C)=O", "NHAc": "NC(C)=O", "NHCOCH3": "NC(C)=O",
    "Ph": "c1ccccc1", "OPh": "Oc1ccccc1", "Bn": "Cc1ccccc1",
    "SMe": "SC", "NMe2": "N(C)C", "N(CH3)2": "N(C)C",
}
_H_VALUES = {"H", "h", "-", "", "H2"}


def _strip_dummies(mol):
    """Delete dummy atom(s); their neighbour picks up an implicit H (used for R=H)."""
    rw = Chem.RWMol(mol)
    for idx in sorted([a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0], reverse=True):
        rw.RemoveAtom(idx)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return m


def _frag_to_attachment(value):
    """R-group text -> attachment fragment '[*:1]<body>' (or None if unparseable)."""
    v = value.strip()
    body = RGROUP_SHORTHAND.get(v) or RGROUP_SHORTHAND.get(v.replace(" ", ""))
    if body is None:
        body = v                          # assume it's already a SMILES fragment
    frag = "[*:1]" + body
    return frag if Chem.MolFromSmiles(frag) is not None else None


def _normalize_scaffold(scaffold):
    """Unify the attachment placeholder to [*:1]."""
    s = scaffold or ""
    for p in ("{R}", "[R]", "[*]", "[*:1]"):
        s = s.replace(p, "[*:1]")
    return s


def enumerate_markush(scaffold, table_text):
    """scaffold : SMILES with exactly ONE attachment point ([*:1] / [*] / [R] / {R}).
    table_text : one 'compound_id  R-group' per line (extra columns after R are ignored).
    Returns list[StructResult] (kind='markush')."""
    scaf_smi = _normalize_scaffold(scaffold)
    scaf = Chem.MolFromSmiles(scaf_smi)
    n_dummy = sum(1 for a in scaf.GetAtoms() if a.GetAtomicNum() == 0) if scaf else 0
    if scaf is None or n_dummy != 1:
        return [StructResult("scaffold", "markush", "failed", raw=scaffold or "",
                note="scaffold must be a valid SMILES with exactly ONE attachment "
                     "point written as [*:1], [*], [R] or {R}.")]

    results = []
    for line in table_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cid = parts[0]
        rval = parts[1] if len(parts) > 1 else ""
        if rval in _H_VALUES:
            smi = _normalize(Chem.MolToSmiles(_strip_dummies(scaf)))
            results.append(StructResult(cid, "markush", "ok" if smi else "failed",
                                        smiles=smi, raw=line,
                                        note="" if smi else "H enumeration failed"))
            continue
        frag = _frag_to_attachment(rval)
        if frag is None:
            results.append(StructResult(cid, "markush", "failed", raw=line,
                                        note=f"could not parse R-group '{rval}'"))
            continue
        try:
            prod = Chem.molzip(scaf, Chem.MolFromSmiles(frag))
            smi = _normalize(Chem.MolToSmiles(prod))
        except Exception:
            smi = None
        results.append(StructResult(cid, "markush", "ok" if smi else "failed",
                                    smiles=smi, raw=line,
                                    note="" if smi else "enumeration failed"))
    return results
