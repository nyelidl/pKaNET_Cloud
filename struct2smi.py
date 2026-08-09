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
            for i, smi in enumerate(_ocsr_backend(file_bytes), start=1):
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
                         note="OCSR backend not installed. Either implement "
                              "_ocsr_backend() with DECIMER/MolScribe, or use the "
                              "manual scaffold+R-group entry in 'Upload PDF / Image'.")]


# -- OCSR plug-in point (implement later) ------------------------------------
def _ocsr_available() -> bool:
    """Return True once a real OCSR engine is wired into _ocsr_backend()."""
    return False

def _ocsr_backend(image_bytes: bytes):                  # pragma: no cover -- stub
    """Plug DECIMER / MolScribe here. Should yield one SMILES per detected
    structure in the image/PDF. Kept out of the import graph so the module stays
    lightweight until an OCSR engine is actually installed."""
    raise NotImplementedError("Wire DECIMER/MolScribe here and yield SMILES.")


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
