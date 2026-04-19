# core.py  —  pKaNET Cloud  (refined)
# Direct port of the Colab notebook pipeline to Streamlit.
# Logic, function signatures, and result keys match the notebook exactly.
#
# ─── REFINEMENT (2026-04) ────────────────────────────────────────────────────
# Fix for polyphenol tautomer misranking (e.g. baicalein, quercetin, catechol,
# pyrogallol): RDKit's TautomerEnumerator produces non-aromatic poly-keto
# tautomers of aromatic polyphenols which the old scorer ranked above the
# correct aromatic form. We now:
#   (a) compare aromatic-ring count of each enumerated tautomer against the
#       input (reference) and apply a hard penalty per ring lost;
#   (b) add SMARTS penalties for the classic traps (pyrogallol-triketo,
#       catechol-diketo, phenol→ring-ketone flips);
#   (c) add a small bonus for phenolic OH preservation.
# All public names (run_job, zip_all_outputs, zip_minimized_structures,
# DISPLAY_COLS, _PKA_BACKEND) are unchanged so app.py keeps working.
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, inchi, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (notebook cell-5 / cell-6 globals)
# ─────────────────────────────────────────────────────────────────────────────
TAUTOMER_PLAUSIBILITY_CUTOFF = 3.0
AMBIGUITY_SCORE_GAP          = 0.5
BORDERLINE_PKA_WINDOW        = 1.0
PUBCHEM_RATE_LIMIT_S         = 0.25
PUBCHEM_CACHE_FILE           = "/tmp/pkanet_pubchem_cache.json"
SEP = "=" * 70

# ─── Aromaticity-guard weights (new, in plausibility-score units) ───────────
# Scale matches existing _BONUS_DEF / _PENALTY_DEF weights (roughly kcal/mol).
W_AROM_RING_LOST         = 8.0   # per aromatic ring that disappears vs input
W_PHENOL_TO_KETO_FLIP    = 6.0   # per phenolic Ar-OH that becomes a ring C=O
W_PYROGALLOL_TRIKETO     = 6.0   # extra penalty for 1,2,3-triketo motif
W_CATECHOL_DIKETO        = 4.0   # extra penalty for 1,2-diketo on sp3 ring
W_PHENOL_PRESERVED_BONUS = 0.5   # small bonus per preserved phenolic OH

# ─────────────────────────────────────────────────────────────────────────────
# Optional dependency probes  (notebook lines 43-88)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _requests = None
    _REQUESTS_OK = False
    print("⚠️  requests not installed — PubChem lookup disabled.")

try:
    from dimorphite_dl import protonate_smiles as _dimorphite_fn
    _DIMORPHITE_OK = True
except ImportError:
    _dimorphite_fn = None
    _DIMORPHITE_OK = False
    print("⚠️  dimorphite-dl not available.")

_PKASOLVER_OK = False
_PROPKA_OK    = False
_UNIPKA_OK    = False
_PKA_BACKEND  = "none"

try:
    from pkasolver.query import QueryModel as _PkaSolverModel  # noqa: F401
    _PKASOLVER_OK = True
    _PKA_BACKEND  = "pkasolver"
    print("✅  pkasolver available.")
except ImportError:
    pass

if not _PKASOLVER_OK:
    try:
        import propka.run as _propka_run  # noqa: F401
        _PROPKA_OK   = True
        _PKA_BACKEND = "propka"
        print("✅  propka available.")
    except ImportError:
        pass

if not _PKASOLVER_OK and not _PROPKA_OK:
    if subprocess.run(["which", "unipka"], capture_output=True).returncode == 0:
        _UNIPKA_OK   = True
        _PKA_BACKEND = "unipka_cli"
        print("✅  unipka CLI available.")

if _PKA_BACKEND == "none":
    print("ℹ️  No ML pKa backend — heuristic ionizable-site table will be used.")

# ─────────────────────────────────────────────────────────────────────────────
# Open Babel helper
# ─────────────────────────────────────────────────────────────────────────────
def check_obabel() -> bool:
    return shutil.which("obabel") is not None


def convert_pdb_to_mol2_obabel(pdb_path: str, mol2_path: str) -> bool:
    if not check_obabel():
        return False
    try:
        r = subprocess.run(
            ["obabel", pdb_path, "-O", mol2_path],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0 and Path(mol2_path).exists()
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────────────────
# STAGE A  ·  RDKit standardization  (notebook lines 100-160)
# ─────────────────────────────────────────────────────────────────────────────

def standardize_smiles(smiles: str) -> tuple[str | None, str]:
    """Validate, normalize, canonicalize. Returns (canonical, status)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, f"❌ RDKit cannot parse: {smiles[:80]}"
    mol = Chem.RemoveHs(mol, implicitOnly=True)
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    try:
        mol = rdMolStandardize.Normalizer().normalize(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True), "OK"


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def smiles_to_inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def enumerate_stereo(smiles: str, keep_original: bool = True) -> list[tuple[str, str | None]]:
    """Return list of (smiles, stereo_tag) pairs."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles[:60]}")
    if keep_original:
        return [(Chem.MolToSmiles(mol, isomericSmiles=True), None)]
    opts = StereoEnumerationOptions(onlyUnassigned=False, unique=True)
    isos = list(EnumerateStereoisomers(mol, options=opts)) or [mol]
    rows: list[tuple[str, str | None]] = []
    for iso in isos:
        smi = Chem.MolToSmiles(iso, isomericSmiles=True)
        tag: str | None = None
        ch  = Chem.FindMolChiralCenters(iso, includeUnassigned=True)
        if len(ch) == 1 and ch[0][1] in ("R", "S"):
            tag = ch[0][1]
        rows.append((smi, tag))
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# STAGE B  ·  PubChem experimental pKa retrieval  (notebook lines 163-390)
# ─────────────────────────────────────────────────────────────────────────────
_PUBCHEM_CACHE: dict = {}


def _load_pubchem_cache() -> None:
    global _PUBCHEM_CACHE
    if Path(PUBCHEM_CACHE_FILE).exists():
        try:
            with open(PUBCHEM_CACHE_FILE) as f:
                _PUBCHEM_CACHE = json.load(f)
        except Exception:
            _PUBCHEM_CACHE = {}


def _save_pubchem_cache() -> None:
    try:
        with open(PUBCHEM_CACHE_FILE, "w") as f:
            json.dump(_PUBCHEM_CACHE, f, indent=2)
    except Exception:
        pass


_load_pubchem_cache()

_PKA_PATTERNS = [
    re.compile(r"pK[aA][\w\s\(\)]*?=\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"([+-]?\d+(?:\.\d+)?)\s*\((?:pK[aA]|acid dissociation)[^)]*\)", re.IGNORECASE),
    re.compile(r"(?:pK[aA]).*?([+-]?\d+(?:\.\d+))", re.IGNORECASE),
]


def _pubchem_get(url: str, timeout: int = 12) -> dict | None:
    if not _REQUESTS_OK or _requests is None:
        return None
    try:
        time.sleep(PUBCHEM_RATE_LIMIT_S)
        r = _requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def pubchem_cid_from_inchikey(inchikey: str) -> int | None:
    key = f"cid:{inchikey}"
    if key in _PUBCHEM_CACHE:
        return _PUBCHEM_CACHE[key]
    data = _pubchem_get(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/cids/JSON"
    )
    cid = None
    if data:
        try:
            cid = int(data["IdentifierList"]["CID"][0])
        except Exception:
            pass
    _PUBCHEM_CACHE[key] = cid
    _save_pubchem_cache()
    return cid


def _flatten_pubchem_section(section: dict, target_heading: str) -> list[str]:
    results: list[str] = []
    if target_heading.lower() in section.get("TOCHeading", "").lower():
        for info in section.get("Information", []):
            for swm in info.get("Value", {}).get("StringWithMarkup", []):
                s = swm.get("String", "").strip()
                if s:
                    results.append(s)
    for sub in section.get("Section", []):
        results.extend(_flatten_pubchem_section(sub, target_heading))
    return results


def pubchem_get_dissociation_texts(cid: int) -> list[str]:
    key = f"diss:{cid}"
    if key in _PUBCHEM_CACHE:
        return _PUBCHEM_CACHE[key]
    data = _pubchem_get(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        "?heading=Dissociation+Constants"
    )
    texts: list[str] = []
    if data:
        try:
            for sec in data.get("Record", {}).get("Section", []):
                texts.extend(_flatten_pubchem_section(sec, "Dissociation"))
        except Exception:
            pass
    _PUBCHEM_CACHE[key] = texts
    _save_pubchem_cache()
    return texts


def parse_pka_values(texts: list[str]) -> tuple[list[float], list[str], dict]:
    full_text = " ".join(texts).lower()
    found: list[float] = []
    src:   list[str]   = []
    for text in texts:
        hits = []
        for pat in _PKA_PATTERNS:
            for m in pat.finditer(text):
                try:
                    v = float(m.group(1))
                    if -5.0 <= v <= 20.0:
                        hits.append(v)
                except ValueError:
                    pass
        if hits:
            found.extend(hits)
            src.append(text)
    dedup: list[float] = []
    for v in found:
        if not any(abs(v - e) < 0.05 for e in dedup):
            dedup.append(v)

    site_labels = bool(re.search(r"pK[aA]\s*[12\(]", " ".join(texts)))
    temperature  = bool(re.search(r"\d+\s*°\s*[Cc]|at\s+\d+\s*[Cc]", full_text))
    solvent      = bool(re.search(r"\b(water|aqueous|etoh|dmso|methanol|buffer|solution)\b", full_text))
    vague        = bool(re.search(
        r"\b(approximately|approx|about|ca\.|around|range|varies|estimated|uncertain|unclear|conflicting)\b",
        full_text))
    conflicting  = (
        len(dedup) >= 2
        and any(abs(a - b) > 1.5 for i, a in enumerate(dedup) for b in dedup[i + 1:])
    )
    if not dedup or conflicting or vague:
        confidence = "low"
    elif len(dedup) > 1 or temperature or solvent or site_labels:
        confidence = "medium"
    else:
        confidence = "high"

    flags = {
        "exact_numeric_match":   bool(dedup),
        "multiple_values_found": len(dedup) > 1,
        "site_labels_found":     site_labels,
        "temperature_mentioned": temperature,
        "solvent_mentioned":     solvent,
        "conflicting_values":    conflicting,
        "vague_or_approximate":  vague,
        "unclear_site_mapping":  len(dedup) > 1,
        "confidence":            confidence,
    }
    return dedup, src, flags


def pubchem_lookup(smiles: str) -> dict:
    """Full PubChem evidence retrieval for a SMILES."""
    result = dict(available=False, cid=None, inchikey=None,
                  pka_values=[], source_texts=[], flags={}, confidence="low", error=None)
    ik = smiles_to_inchikey(smiles)
    if ik is None:
        result["error"] = "InChIKey computation failed."
        return result
    result["inchikey"] = ik

    cid = pubchem_cid_from_inchikey(ik)
    if cid is None:
        result["error"] = "CID not found."
        return result
    result["cid"] = cid

    texts = pubchem_get_dissociation_texts(cid)
    if not texts:
        result["error"] = "No dissociation constant data on PubChem."
        return result

    vals, srcs, flags = parse_pka_values(texts)
    result.update(available=bool(vals), pka_values=vals,
                  source_texts=srcs, flags=flags,
                  confidence=flags.get("confidence", "low"))
    return result

# ─────────────────────────────────────────────────────────────────────────────
# STAGE C  ·  ML pKa backends  (notebook lines 392-510)
# ─────────────────────────────────────────────────────────────────────────────

def _unipka_via_pkasolver(smiles: str) -> list[dict]:
    try:
        from pkasolver.query import QueryModel
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        df = QueryModel().predict_pka(mol)
        return [{"pka": float(row.get("pKa", row.get("pka", 0))),
                 "site_type": str(row.get("type", "?")),
                 "site_label": str(row.get("atom_idx", "?")),
                 "source": "pkasolver", "confidence": "ml_gnn"}
                for _, row in df.iterrows()]
    except Exception as e:
        print(f"⚠️  pkasolver failed: {e}")
        return []


def _unipka_via_propka(smiles: str) -> list[dict]:
    try:
        import propka.run as pk
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        mol = Chem.AddHs(mol)
        p = AllChem.ETKDGv3(); p.randomSeed = 42
        if AllChem.EmbedMolecule(mol, p) != 0:
            return []
        AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tf:
            tmppath = tf.name
            tf.write(Chem.MolToPDBBlock(mol))
        results = []
        try:
            mc = pk.single(tmppath, optargs=["--quiet"])
            for grp in mc.conformations[0].groups:
                pv = getattr(grp, "pka_value", None)
                if pv is not None:
                    results.append({"pka": float(pv),
                                    "site_label": str(getattr(grp, "atom_name", "?")),
                                    "site_type":  str(getattr(grp, "type", "?")),
                                    "source": "propka", "confidence": "semi_empirical"})
        finally:
            try:
                os.unlink(tmppath)
            except Exception:
                pass
        return results
    except Exception as e:
        print(f"⚠️  propka failed: {e}")
        return []


def _unipka_via_cli(smiles: str) -> list[dict]:
    try:
        r = subprocess.run(["unipka", "--smiles", smiles, "--json"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return []
        data = json.loads(r.stdout)
        return [{"pka": e.get("pka"), "site_label": e.get("site", "?"),
                 "site_type": e.get("type", "?"), "source": "unipka_cli", "confidence": "ml"}
                for e in data.get("microstates", [])]
    except Exception as e:
        print(f"⚠️  unipka CLI failed: {e}")
        return []


def unipka_predict(smiles: str) -> list[dict]:
    """Try ML pKa backends in priority order; return full site-prediction list."""
    if _UNIPKA_OK:
        r = _unipka_via_cli(smiles)
        if r:
            return r
    if _PKASOLVER_OK:
        r = _unipka_via_pkasolver(smiles)
        if r:
            return r
    if _PROPKA_OK:
        r = _unipka_via_propka(smiles)
        if r:
            return r
    return []


def unipka_summary_pka(predictions: list[dict]) -> tuple[float | None, str]:
    """Lossy scalar summary — display only; do NOT use for ranking."""
    valid = [p for p in predictions if p.get("pka") is not None]
    if not valid:
        return None, "none"
    closest = min(valid, key=lambda p: abs(float(p["pka"]) - 7.4))
    return float(closest["pka"]), closest.get("source", "?")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE D  ·  Dimorphite-DL protonation enumerator  (notebook lines 517-570)
# ─────────────────────────────────────────────────────────────────────────────

def dimorphite_enumerate(
    smiles: str,
    ph_min: float,
    ph_max: float,
    precision: float = 1.0,
    max_variants: int = 128,
) -> list[str]:
    """Enumerate protonation states; always includes the input SMILES."""
    if not _DIMORPHITE_OK or _dimorphite_fn is None:
        return [smiles]

    kwarg_variants = [
        {"ph_min": ph_min, "ph_max": ph_max, "precision":     precision, "max_variants": max_variants},
        {"min_ph": ph_min, "max_ph": ph_max, "pka_precision": precision, "max_variants": max_variants},
        {"ph_min": ph_min, "ph_max": ph_max, "precision":     precision},
        {"min_ph": ph_min, "max_ph": ph_max, "pka_precision": precision},
    ]
    errors: list[str] = []
    raw:    list[str] = []
    for kwargs in kwarg_variants:
        try:
            r   = _dimorphite_fn(smiles, **kwargs)
            raw = [r] if isinstance(r, str) else list(r or [])
            if raw:
                break
        except TypeError as e:
            errors.append(str(e))

    if not raw:
        try:
            sig = inspect.signature(_dimorphite_fn)
            kw: dict = {}
            for name in sig.parameters:
                lo = name.lower()
                if   lo in {"ph_min", "min_ph"}:           kw[name] = ph_min
                elif lo in {"ph_max", "max_ph"}:           kw[name] = ph_max
                elif lo in {"precision", "pka_precision"}: kw[name] = precision
                elif lo == "max_variants":                 kw[name] = max_variants
            r   = _dimorphite_fn(smiles, **kw)
            raw = [r] if isinstance(r, str) else list(r or [])
        except Exception as e:
            errors.append(str(e))
            print(f"⚠️  dimorphite-dl failed ({smiles[:50]}). Errors: {errors[-2:]}")

    seen:   set[str]  = set()
    result: list[str] = []
    seed = canonicalize(smiles)
    if seed:
        seen.add(seed); result.append(seed)
    for smi in raw:
        c = canonicalize(smi)
        if c and c not in seen:
            seen.add(c); result.append(c)
    return result or [smiles]

# ─────────────────────────────────────────────────────────────────────────────
# STAGES E + F  ·  HH scoring + chemistry filters  (notebook lines 573-765)
# ─────────────────────────────────────────────────────────────────────────────

def hh_fraction_charged(pka: float, ph: float, site_type: str) -> float:
    if site_type == "acid":
        return 1.0 / (1.0 + 10.0 ** (pka - ph))
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


def hh_ph_match_score(pka: float, ph: float, site_type: str, actual_charge: int) -> float:
    f_charged = hh_fraction_charged(pka, ph, site_type)
    dpH       = abs(ph - pka)
    if site_type == "acid":
        expected_neg = f_charged > 0.5
        if expected_neg and actual_charge < 0:   return  min(1.2, dpH * 0.45)
        elif expected_neg:                        return -min(1.0, dpH * 0.35)
        elif actual_charge >= 0:                  return  0.1
        else:                                     return -min(1.2, dpH * 0.40)
    else:
        expected_pos = f_charged > 0.5
        if expected_pos and actual_charge > 0:    return  min(1.2, dpH * 0.45)
        elif expected_pos:                        return -min(1.0, dpH * 0.35)
        elif actual_charge <= 0:                  return  0.1
        else:                                     return -min(1.2, dpH * 0.40)


_IONIZABLE_SITE_DEF = [
    ("sulfonic_acid",      "[SX4](=O)(=O)[OX2H1]",                              1.0,  "acid"),
    ("phosphoric_mono",    "[PX4](=O)([OX2H1])([OX2H1])[OX2H1]",               2.1,  "acid"),
    ("carboxylic_acid",    "[CX3](=O)[OX2H1]",                                  4.5,  "acid"),
    ("tetrazole",          "c1nn[nH]n1",                                         4.9,  "acid"),
    ("imidazole",          "c1cn[nH]c1",                                         6.0,  "acid"),
    ("benzimidazole",      "c1ccc2[nH]cnc2c1",                                  5.5,  "acid"),
    ("phosphonate",        "[PX4](=O)([OX2H1])[OX2H1,OX1-]",                   6.5,  "acid"),
    ("sulfonamide_NH",     "[SX4](=O)(=O)[NX3;H1]",                            10.1,  "acid"),
    ("imide_NH",           "[CX3](=O)[NX3;H1][CX3]=O",                          9.6,  "acid"),
    ("acylhydrazone_NH",   "[CX3](=O)[NX3;H1][NX2]=[CX3]",                    10.5,  "acid"),
    ("hydrazide_NH",       "[CX3](=O)[NX3;H1][NX3;H2]",                        10.5,  "acid"),
    ("urea_NH",            "[NX3;H1][CX3](=O)[NX3;H1,H2]",                     13.0,  "acid"),
    ("amide_NH",           "[CX3](=O)[NX3;H1,H2;!$([N]~N)]",                   15.0,  "acid"),
    ("phenol",             "c[OX2H1]",                                          10.0,  "acid"),
    ("thiol_arom",         "c[SX2H1]",                                           6.5,  "acid"),
    ("thiol_aliph",        "[CX4][SX2H1]",                                      10.5,  "acid"),
    ("aniline",            "c[NX3;H1,H2;!$(N~[!#6])]",                          4.6,  "base"),
    ("pyridine_like",      "[$([nX2]1:[c,n]:c:[c,n]:c1),$([nX2]:c:n)]",         5.2,  "base"),
    ("aliphatic_amine",    "[NX3;H1,H2;!$(NC=O);!$(N~[!#6;!H]);!$([nH])]",      9.5,  "base"),
    ("aliphatic_amine_t",  "[NX3;H0;!$(NC=O);!$(Nc);!$([nH]);!$([N]~[!#6])]",   9.0,  "base"),
    ("amidine",            "[CX3](=[NX2;H0,H1])[NX3;H1,H2]",                   12.4,  "base"),
    ("guanidine",          "[NX3][CX3](=[NX2])[NX3]",                           13.0,  "base"),
]

_IONIZABLE_SITES_COMPILED: list[tuple] = []
for _lbl, _sma, _pka_v, _typ in _IONIZABLE_SITE_DEF:
    _pat = Chem.MolFromSmarts(_sma)
    if _pat is not None:
        _IONIZABLE_SITES_COMPILED.append((_lbl, _pat, _pka_v, _typ))
    else:
        print(f"⚠️  SMARTS compile failed: {_lbl}")


def find_ionizable_sites(mol: Chem.Mol) -> list[dict]:
    sites:  list[dict]     = []
    seen_k: set[frozenset] = set()
    for lbl, pat, pka_v, stype in _IONIZABLE_SITES_COMPILED:
        for match in mol.GetSubstructMatches(pat):
            k = frozenset(match)
            if k in seen_k:
                continue
            seen_k.add(k)
            sites.append(dict(label=lbl, atom_indices=list(match),
                               heuristic_pka=pka_v, site_type=stype))
    return sites


_BONUS_DEF = [
    ("amide",            +2.5, "[CX3](=O)[NX3;H1,H2]"),
    ("lactam",           +2.5, "[C;R](=O)[N;R]"),
    ("acylhydrazone_NH", +2.0, "[CX3](=O)[NX3;H1][NX2]=[CX3]"),
    ("hydrazide_NH",     +2.0, "[CX3](=O)[NX3;H1][NX3;H2]"),
    ("urea_NH",          +1.5, "[NX3;H1][CX3](=O)[NX3;H1,H2]"),
    ("thioamide",        +1.0, "[CX3](=S)[NX3;H1,H2]"),
    ("aromatic_ring",    +0.3, "c1ccccc1"),
    # ── NEW: phenol preservation bonus (small, per phenolic OH) ─────────────
    ("phenol_preserved", W_PHENOL_PRESERVED_BONUS, "c[OX2H1]"),
]
_PENALTY_DEF = [
    ("imidic_acid_open", -4.0, "[CX3;!R](=[NX2])[OX2H1]"),
    ("lactim_ring",      -4.0, "[C;R](=[NX2])[OX2H1]"),
    ("iminol_general",   -3.5, "[NX2]=[CX3][OX2H1]"),
    ("amide_N_deproton", -5.0, "[$([NX3-]C=O),$([NX3-]c=O)]"),
    ("enol_simple",      -1.2, "[CX3](=[CX3])[OX2H1]"),
    # ── NEW: aromaticity-destroying keto-tautomer traps ─────────────────────
    # Pyrogallol → 1,2,3-triketo cyclohexane/ene (baicalein, gallic acid …)
    ("pyrogallol_triketo", -W_PYROGALLOL_TRIKETO,
        "[#6;!a;R]1(=O)[#6;!a;R](=O)[#6;!a;R](=O)[#6;R][#6;R][#6;R]1"),
    # Catechol → 1,2-diketo on a non-aromatic six-membered ring
    ("catechol_diketo",    -W_CATECHOL_DIKETO,
        "[#6;!a;R]1(=O)[#6;!a;R](=O)[#6;R][#6;R][#6;R][#6;R]1"),
    # Ortho para-quinone forms of phenols (loss of aromaticity on 6-ring)
    ("ring_carbonyl_onaromring_former", -3.0,
        "[#6;!a;R](=O)[#6;!a;R]=[#6;!a;R]"),
]
_CHEM_RULES: list[tuple[str, float, Chem.Mol]] = []
for _lbl, _wt, _sma in _BONUS_DEF + _PENALTY_DEF:
    _pat = Chem.MolFromSmarts(_sma)
    if _pat is not None:
        _CHEM_RULES.append((_lbl, _wt, _pat))
    else:
        print(f"⚠️  SMARTS compile failed: {_lbl}")

_TAUTOMER_RICH_DEF = [
    ("imidazole",    "[nH]1ccnc1"),
    ("benzimidazole","c1ccc2[nH]cnc2c1"),
    ("tetrazole",    "c1nn[nH]n1"),
    ("triazole",     "[nH]1ccnn1"),
    ("pyridone",     "[OH]c1ccccn1"),
    ("keto_enol",    "[CX4][CX3](=O)[CX4]"),
    ("purine",       "c1ncnc2[nH]cnc12"),
]
_TAUTOMER_RICH_COMPILED = [
    (lbl, pat)
    for lbl, sma in _TAUTOMER_RICH_DEF
    if (pat := Chem.MolFromSmarts(sma)) is not None
]


# ─── Aromaticity comparison helper (NEW) ─────────────────────────────────────
def _n_aromatic_rings(mol: Chem.Mol | None) -> int:
    if mol is None:
        return 0
    try:
        return int(rdMolDescriptors.CalcNumAromaticRings(mol))
    except Exception:
        return 0


def _count_phenolic_OH(mol: Chem.Mol | None) -> int:
    if mol is None:
        return 0
    patt = Chem.MolFromSmarts("c[OX2H1]")
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt))


def score_tautomer_plausibility(
    smiles: str,
    ref_mol: Chem.Mol | None = None,
) -> tuple[float, dict]:
    """Score a tautomer. If ref_mol is given, also penalize aromatic-ring loss
    and phenol→ring-ketone flips relative to the reference (input) molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -999.0, {}
    bd: dict[str, float] = {}
    total = 0.0

    # Static SMARTS bonuses / penalties (existing + new traps)
    for lbl, wt, pat in _CHEM_RULES:
        n = len(mol.GetSubstructMatches(pat))
        if n:
            c = wt * n
            bd[lbl] = round(c, 3)
            total  += c

    # ── NEW: reference-aware aromaticity & phenol-preservation penalties ────
    if ref_mol is not None:
        n_arom_ref  = _n_aromatic_rings(ref_mol)
        n_arom_taut = _n_aromatic_rings(mol)
        rings_lost  = max(0, n_arom_ref - n_arom_taut)
        if rings_lost > 0:
            pen = -W_AROM_RING_LOST * rings_lost
            bd["arom_ring_lost_vs_input"] = round(pen, 3)
            total += pen

        n_phenol_ref  = _count_phenolic_OH(ref_mol)
        n_phenol_taut = _count_phenolic_OH(mol)
        phenols_lost  = max(0, n_phenol_ref - n_phenol_taut)
        if phenols_lost > 0:
            pen = -W_PHENOL_TO_KETO_FLIP * phenols_lost
            bd["phenol_flipped_to_keto"] = round(pen, 3)
            total += pen

    bd["_total"] = round(total, 3)
    return total, bd


def is_tautomer_rich(mol: Chem.Mol) -> tuple[bool, list[str]]:
    hits = [l for l, p in _TAUTOMER_RICH_COMPILED if mol.HasSubstructMatch(p)]
    return bool(hits), hits


def enumerate_and_filter_tautomers(
    smiles: str,
    max_states: int = 8,
    cutoff: float   = TAUTOMER_PLAUSIBILITY_CUTOFF,
) -> tuple[list[dict], list[dict], bool, list[str]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles[:60]}")

    # Reference molecule for aromaticity/phenol comparison (the INPUT).
    ref_mol = mol

    tr_flag, tr_motifs = is_tautomer_rich(mol)
    enum  = rdMolStandardize.TautomerEnumerator()
    seen: set[str] = set()
    scored: list[dict] = []

    # Always include the input canonical form in the pool so it can win even if
    # the enumerator reorders or omits it.
    input_canon = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    seen.add(input_canon)
    sc0, bd0 = score_tautomer_plausibility(input_canon, ref_mol=ref_mol)
    scored.append({"smiles": input_canon, "score": sc0, "breakdown": bd0})

    for tmol in enum.Enumerate(mol):
        smi = Chem.MolToSmiles(tmol, isomericSmiles=True, canonical=True)
        if smi in seen:
            continue
        seen.add(smi)
        sc, bd = score_tautomer_plausibility(smi, ref_mol=ref_mol)
        scored.append({"smiles": smi, "score": sc, "breakdown": bd})

    if not scored:
        smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        sc, bd = score_tautomer_plausibility(smi, ref_mol=ref_mol)
        scored = [{"smiles": smi, "score": sc, "breakdown": bd}]

    scored     = sorted(scored, key=lambda x: -x["score"])[:max_states]
    best       = scored[0]["score"]
    eff_cutoff = cutoff * (2.0 if tr_flag else 1.0)
    kept      = [t for t in scored if t["score"] >= best - eff_cutoff]
    discarded = [t for t in scored if t["score"] <  best - eff_cutoff]
    return kept or [scored[0]], discarded, tr_flag, tr_motifs

# ─────────────────────────────────────────────────────────────────────────────
# STAGE G  ·  Microstate generation, scoring, ranking  (notebook lines 768-1097)
# ─────────────────────────────────────────────────────────────────────────────

def get_charge_profile(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles[:60]}")
    net = n_pos = n_neg = 0
    rows: list[dict] = []
    for atom in mol.GetAtoms():
        fc = int(atom.GetFormalCharge())
        net  += fc
        n_pos += fc > 0
        n_neg += fc < 0
        if fc != 0:
            rows.append({"atom_idx": atom.GetIdx(), "symbol": atom.GetSymbol(),
                          "formal_charge": fc})
    return {
        "net_charge":           int(net),
        "n_pos_atoms":          int(n_pos),
        "n_neg_atoms":          int(n_neg),
        "has_pos":              n_pos > 0,
        "has_neg":              n_neg > 0,
        "is_zwitterion_strict": bool(n_pos > 0 and n_neg > 0 and net == 0),
        "charged_atoms":        rows,
    }


def charged_atoms_text(cp: dict) -> str:
    rows = cp.get("charged_atoms", [])
    if not rows:
        return "none"
    return ", ".join(f"{r['symbol']}{r['atom_idx']}({r['formal_charge']:+d})" for r in rows)


def _best_pka_for_site(site: dict, ml_predictions: list[dict], pubchem_result: dict) -> tuple[float, str]:
    stype = site["site_type"]
    for mp in ml_predictions:
        if mp.get("pka") is not None and mp.get("site_type", "").lower() == stype:
            return float(mp["pka"]), mp.get("source", "ml")
    if pubchem_result.get("available") and pubchem_result.get("confidence") in ("high", "medium"):
        vals = pubchem_result.get("pka_values", [])
        if vals:
            return min(vals, key=lambda v: abs(v - site["heuristic_pka"])), "pubchem"
    return site["heuristic_pka"], "heuristic"


def _label_decision_backend(
    ml_predictions: list[dict],
    pubchem_result: dict,
    used_heuristic: bool,
) -> tuple[str, str]:
    has_ml = bool(ml_predictions)
    has_pc = pubchem_result.get("available", False)
    ml_src = ml_predictions[0].get("source", "ml") if has_ml else None

    if has_ml and has_pc:
        ml_vals = [p["pka"] for p in ml_predictions if p.get("pka") is not None]
        pc_vals = pubchem_result.get("pka_values", [])
        if ml_vals and pc_vals:
            avg_diff = abs(sum(ml_vals)/len(ml_vals) - sum(pc_vals)/len(pc_vals))
            backend  = f"{ml_src}_pubchem_consistent" if avg_diff <= 1.5 else "mixed_evidence"
        else:
            backend = f"{ml_src}_supported"
        mode = "ml_pka_dominant"
    elif has_ml:
        backend = f"{ml_src}_supported"
        mode    = "ml_pka_dominant"
    elif has_pc:
        backend = "pubchem_supported"
        mode    = "pubchem_pka_dominant"
    else:
        backend = "heuristic_only"
        mode    = "heuristic_only"
    return backend, mode


def score_microstate_full(
    microstate_smiles: str,
    tautomer_smiles:   str,
    taut_plausibility: float,
    taut_breakdown:    dict,
    ion_sites:         list[dict],
    ml_predictions:    list[dict],
    pubchem_result:    dict,
    target_ph:         float,
    ref_mol:           Chem.Mol | None = None,
) -> tuple[float, dict, dict, bool]:
    mol = Chem.MolFromSmiles(microstate_smiles)
    if mol is None:
        return -1e9, {}, {}, False

    cp  = get_charge_profile(microstate_smiles)
    net = cp["net_charge"]
    n_pos, n_neg = cp["n_pos_atoms"], cp["n_neg_atoms"]
    fc_map = {a.GetIdx(): a.GetFormalCharge() for a in mol.GetAtoms()}

    # Layer 1 — safety
    pat_amide_neg = Chem.MolFromSmarts("[$([NX3-]C=O),$([NX3-]c=O)]")
    n_amide_neg   = len(mol.GetSubstructMatches(pat_amide_neg)) if pat_amide_neg else 0
    s_amide_n_dep = -5.0 * n_amide_neg

    # Layer 1b — aromaticity-loss propagation to microstate (NEW)
    # If the protonated microstate itself lost aromatic rings vs the input,
    # propagate that penalty here too — protonation changes can mask it.
    s_arom_loss = 0.0
    if ref_mol is not None:
        rings_lost = max(0, _n_aromatic_rings(ref_mol) - _n_aromatic_rings(mol))
        if rings_lost > 0:
            s_arom_loss = -W_AROM_RING_LOST * rings_lost

    # Layer 2 — tautomer plausibility
    s_tautomer = 0.65 * taut_plausibility

    # Layer 3 — pH consistency (HH)
    borderline = False
    ph_bd: dict[str, float] = {}
    s_ph = 0.0
    for site in ion_sites:
        pka_val, pka_src = _best_pka_for_site(site, ml_predictions, pubchem_result)
        if abs(target_ph - pka_val) <= BORDERLINE_PKA_WINDOW:
            borderline = True
        site_charge = sum(fc_map.get(i, 0) for i in site["atom_indices"])
        contrib     = hh_ph_match_score(pka_val, target_ph, site["site_type"], site_charge)
        ph_bd[f"pH_{site['label']}[{pka_src}]"] = round(contrib, 3)
        s_ph += contrib

    # Layer 4 — PubChem evidence bonus
    s_pubchem_bonus = 0.0
    if pubchem_result.get("available"):
        pc_weight = {"high": 1.0, "medium": 0.6, "low": 0.2}.get(
            pubchem_result.get("confidence", "low"), 0.2)
        for pka_val in pubchem_result["pka_values"]:
            exp = -1 if hh_fraction_charged(pka_val, target_ph, "acid") > 0.5 else 0
            s_pubchem_bonus += 0.25 * pc_weight if net == exp else -0.15 * pc_weight
        s_pubchem_bonus = max(-0.4, min(0.5, s_pubchem_bonus))

    # Layer 5 — charge-structure reasonableness
    has_acid_site = any(s["site_type"] == "acid" and (target_ph - s["heuristic_pka"]) > 1.0
                        for s in ion_sites)
    has_base_site = any(s["site_type"] == "base" and (s["heuristic_pka"] - target_ph) > 1.0
                        for s in ion_sites)
    if cp["is_zwitterion_strict"]:
        s_zwit = 0.8 if (has_acid_site and has_base_site) else -0.6
    else:
        s_zwit = -0.4 if (has_acid_site and has_base_site and net == 0 and n_pos == 0) else 0.0

    strong_acid  = [s for s in ion_sites if s["site_type"] == "acid" and (target_ph - s["heuristic_pka"]) > 2.0]
    strong_base  = [s for s in ion_sites if s["site_type"] == "base" and (s["heuristic_pka"] - target_ph) > 2.0]
    s_improbable = 0.0
    if strong_acid and net >= 0 and n_neg == 0:
        s_improbable -= 0.5 * len(strong_acid)
    if strong_base and net <= 0 and n_pos == 0:
        s_improbable -= 0.5 * len(strong_base)
    s_multi = -0.12 * max(0, n_pos + n_neg - 2)

    total = (s_amide_n_dep + s_arom_loss + s_tautomer + s_ph
             + s_pubchem_bonus + s_zwit + s_improbable + s_multi)

    def _has_key(bd: dict, keys: list[str], positive: bool) -> bool:
        return any(bd.get(k, 0) * (1 if positive else -1) > 0 for k in keys)

    flag_amide  = _has_key(taut_breakdown, ["amide","lactam","acylhydrazone_NH","hydrazide_NH"], True)
    flag_imidic = _has_key(taut_breakdown, ["imidic_acid_open","lactim_ring","iminol_general"], False)
    flag_lactim = taut_breakdown.get("lactim_ring", 0) < 0
    flag_arom_lost = s_arom_loss < 0 or taut_breakdown.get("arom_ring_lost_vs_input", 0) < 0
    used_heuristic = not bool(ml_predictions) and not pubchem_result.get("available")
    decision_backend, decision_mode = _label_decision_backend(
        ml_predictions, pubchem_result, used_heuristic)

    cp.update(
        flag_amide_preserved               = flag_amide,
        flag_imidic_acid_penalty           = flag_imidic,
        flag_lactim_penalty                = flag_lactim,
        flag_amide_n_deprotonation_penalty = n_amide_neg > 0,
        flag_aromaticity_lost              = flag_arom_lost,
        decision_backend                   = decision_backend,
        decision_mode                      = decision_mode,
    )
    bd_full = {
        "s_amide_n_deproton [safety]": round(s_amide_n_dep,   3),
        "s_aromaticity_loss [safety]": round(s_arom_loss,     3),
        "s_tautomer_plausibility":     round(s_tautomer,      3),
        "s_ph_consistency [HH]":       round(s_ph,            3),
        "s_pubchem_evidence_bonus":    round(s_pubchem_bonus, 3),
        "s_zwitterion_consistency":    round(s_zwit,          3),
        "s_improbable_neutral":        round(s_improbable,    3),
        "s_multicharge_penalty":       round(s_multi,         3),
        "total_score":                 round(total,           3),
        **ph_bd,
    }
    return total, cp, bd_full, borderline


def generate_ranked_microstates(
    base_smiles:    str,
    target_ph:      float = 7.4,
    ph_window:      float = 1.0,
    max_tautomers:  int   = 8,
    top_n:          int   = 5,
    pubchem_result: dict | None = None,
) -> tuple[list[dict], bool, list[dict], bool, list[str], list[dict]]:
    """Full pipeline for one SMILES.

    Returns
    -------
    top_microstates, ambiguous_flag, all_microstates,
    tautomer_rich_flag, tr_motifs, ml_predictions
    """
    if pubchem_result is None:
        pubchem_result = {}

    # Reference molecule used by aromaticity-guard scoring at every layer.
    ref_mol = Chem.MolFromSmiles(base_smiles)

    kept, disc, tr_flag, tr_motifs = enumerate_and_filter_tautomers(
        base_smiles, max_states=max_tautomers, cutoff=TAUTOMER_PLAUSIBILITY_CUTOFF)
    if disc:
        print(f"   🔬  Discarded {len(disc)} implausible tautomers "
              f"(e.g. score={disc[0]['score']:.1f}: {disc[0]['smiles'][:55]})")

    ml_preds  = unipka_predict(base_smiles)
    ion_sites = find_ionizable_sites(ref_mol) if ref_mol else []

    all_micro: list[dict] = []
    seen_smi:  set[str]  = set()

    ph_lo = max(0.0,  target_ph - ph_window / 2)
    ph_hi = min(14.0, target_ph + ph_window / 2)

    for ti, taut in enumerate(kept, 1):
        for pi, psmi in enumerate(dimorphite_enumerate(taut["smiles"], ph_lo, ph_hi), 1):
            if psmi in seen_smi:
                continue
            seen_smi.add(psmi)
            try:
                sc, cp, bd, bl = score_microstate_full(
                    microstate_smiles  = psmi,
                    tautomer_smiles    = taut["smiles"],
                    taut_plausibility  = taut["score"],
                    taut_breakdown     = taut["breakdown"],
                    ion_sites          = ion_sites,
                    ml_predictions     = ml_preds,
                    pubchem_result     = pubchem_result,
                    target_ph          = target_ph,
                    ref_mol            = ref_mol,
                )
            except Exception as e:
                print(f"⚠️  Scoring error ({psmi[:40]}): {e}")
                continue

            all_micro.append({
                "tautomer_rank":                     ti,
                "protomer_rank_in_tautomer":         pi,
                "tautomer_smiles":                   taut["smiles"],
                "tautomer_plausibility":             round(taut["score"], 3),
                "microstate_smiles":                 psmi,
                "parent_smiles":                     base_smiles,
                "selection_score":                   float(sc),
                "net_charge":                        cp["net_charge"],
                "has_pos":                           cp["has_pos"],
                "has_neg":                           cp["has_neg"],
                "is_zwitterion_strict":              cp["is_zwitterion_strict"],
                "charged_atoms":                     charged_atoms_text(cp),
                "charged_atom_rows":                 cp["charged_atoms"],
                "decision_backend":                  cp.get("decision_backend",  "unknown"),
                "decision_mode":                     cp.get("decision_mode",     "unknown"),
                "flag_amide_preserved":              cp.get("flag_amide_preserved",              False),
                "flag_imidic_acid_penalty":          cp.get("flag_imidic_acid_penalty",          False),
                "flag_lactim_penalty":               cp.get("flag_lactim_penalty",               False),
                "flag_amide_n_deprotonation_penalty":cp.get("flag_amide_n_deprotonation_penalty",False),
                "flag_aromaticity_lost":             cp.get("flag_aromaticity_lost",             False),
                "flag_borderline_pka":               bl,
                "flag_tautomer_rich":                tr_flag,
                "flag_pubchem_text_ambiguous":       pubchem_result.get("flags",{}).get("vague_or_approximate", False),
                "flag_pubchem_conflicting":          pubchem_result.get("flags",{}).get("conflicting_values",   False),
                "flag_pubchem_confidence":           pubchem_result.get("confidence", "n/a"),
                "flag_unipka_used":                  bool(ml_preds),
                "flag_dimorphite_used":              _DIMORPHITE_OK,
                "pKa_source": (ml_preds[0]["source"] if ml_preds else
                               ("pubchem" if pubchem_result.get("available") else "heuristic")),
                **{f"score_{k}": v for k, v in bd.items()},
                **{f"taut_{k}":  v for k, v in taut["breakdown"].items()},
            })

    if not all_micro:
        return [], False, [], tr_flag, tr_motifs, ml_preds

    all_micro.sort(key=lambda x: (
        -x["selection_score"], abs(x["net_charge"]), x["tautomer_rank"], x["microstate_smiles"]
    ))
    best_sc = all_micro[0]["selection_score"]

    for i, row in enumerate(all_micro, 1):
        row["microstate_rank"] = i
        row["delta_from_best"] = round(best_sc - row["selection_score"], 3)

    top = all_micro[:max(1, top_n)]

    score_ambig = len(top) > 1 and top[1]["delta_from_best"] <= AMBIGUITY_SCORE_GAP
    ambiguous   = score_ambig or any(r["flag_borderline_pka"] for r in top[:2]) or tr_flag

    for row in all_micro:
        row["ambiguous_top_assignment"] = ambiguous
        row["flag_multiprotic"]         = len(ion_sites) >= 2

    return top, ambiguous, all_micro, tr_flag, tr_motifs, ml_preds

# ─────────────────────────────────────────────────────────────────────────────
# STAGE H  ·  3D construction  (notebook lines 1099-1125)
# ─────────────────────────────────────────────────────────────────────────────

def build_minimized_3d(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles[:60]}")
    mol = Chem.AddHs(mol)
    p   = AllChem.ETKDGv3(); p.randomSeed = 42
    if AllChem.EmbedMolecule(mol, p) != 0:
        raise ValueError("ETKDG embedding failed.")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    return mol


def mol_from_file(filepath: str) -> Chem.Mol:
    ext = os.path.splitext(filepath)[1].lower()
    mol = None
    if ext == ".sdf":
        mol = next((m for m in Chem.SDMolSupplier(filepath, removeHs=False) if m), None)
    elif ext == ".mol2":
        mol = Chem.MolFromMol2File(filepath, removeHs=False)
    elif ext == ".pdb":
        mol = Chem.MolFromPDBFile(filepath, removeHs=False)
    if mol is None:
        raise ValueError(f"Cannot parse: {filepath}")
    return mol

# ─────────────────────────────────────────────────────────────────────────────
# Display columns  (notebook line 1149)
# ─────────────────────────────────────────────────────────────────────────────
DISPLAY_COLS = [
    "microstate_rank", "tautomer_rank", "tautomer_plausibility",
    "microstate_smiles", "selection_score", "net_charge", "charged_atoms",
    "decision_backend", "decision_mode",
    "flag_amide_preserved", "flag_imidic_acid_penalty",
    "flag_amide_n_deprotonation_penalty", "flag_aromaticity_lost",
    "flag_borderline_pka",
    "flag_pubchem_text_ambiguous", "flag_unipka_used", "pKa_source",
    "delta_from_best",
]

# ─────────────────────────────────────────────────────────────────────────────
# File helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_smi_lines(text: str) -> list[tuple[str, str]]:
    records = []
    idx = 1
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        records.append((parts[0], parts[1] if len(parts) > 1 else f"mol_{idx:03d}"))
        idx += 1
    return records


def save_2d_image(smiles: str, path: str, size: tuple = (800, 600)) -> bool:
    try:
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        AllChem.Compute2DCoords(mol)
        Draw.MolToImage(mol, size=size).save(path)
        return True
    except Exception:
        return False


def save_molecule_files(mol: Chem.Mol, base_path: str, formats: list[str]) -> dict:
    """Save SDF (always) + user-requested PDB/MOL2. Returns dict with file paths + warnings list."""
    saved:    dict      = {}
    warnings: list[str] = []
    mol2_via_obabel = False

    try:
        sdf_path = f"{base_path}.sdf"
        w = Chem.SDWriter(sdf_path); w.write(mol); w.close()
        saved["sdf"] = sdf_path
    except Exception as e:
        warnings.append(f"Could not save SDF: {e}")

    for fmt in [f.upper() for f in formats]:
        if fmt == "SDF":
            continue
        try:
            if fmt == "PDB":
                fp = f"{base_path}.pdb"
                Chem.MolToPDBFile(mol, fp)
                saved["pdb"] = fp
            elif fmt == "MOL2":
                fp = f"{base_path}.mol2"
                if hasattr(Chem, "MolToMol2File"):
                    try:
                        Chem.MolToMol2File(mol, fp)
                        saved["mol2"] = fp
                        continue
                    except Exception:
                        pass
                if "pdb" not in saved:
                    pdb_fp = f"{base_path}.pdb"
                    Chem.MolToPDBFile(mol, pdb_fp)
                    saved["pdb"] = pdb_fp
                if convert_pdb_to_mol2_obabel(saved["pdb"], fp):
                    saved["mol2"] = fp
                    mol2_via_obabel = True
                else:
                    warnings.append(
                        "MOL2 unavailable — install Open Babel (obabel)."
                        if not check_obabel() else "MOL2 conversion failed."
                    )
        except Exception as e:
            warnings.append(f"Could not save {fmt}: {e}")

    if mol2_via_obabel:
        warnings.append("ℹ️ MOL2 generated via Open Babel (converted from PDB)")

    saved["warnings"] = warnings
    return saved

# ─────────────────────────────────────────────────────────────────────────────
# run_job  —  Streamlit adapter for the notebook MAIN WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────

def run_job(
    *,
    input_type:              str,
    smiles_text:             str | None,
    uploaded_bytes:          bytes | None,
    uploaded_name:           str | None,
    target_pH:               float,
    output_name:             str,
    out_dir:                 str,
    output_formats:          list[str] | None = None,
    enumerate_stereoisomers: bool  = True,
    use_pubchem:             bool  = True,
    ph_window:               float = 1.0,
    max_tautomers:           int   = 8,
    top_n_microstates:       int   = 5,
    write_alt_3d_for_top_k:  int   = 3,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not output_formats:
        output_formats = ["PDB"]

    # ── Collect raw ligands ──────────────────────────────────────────────────
    ligands_raw: list[dict] = []

    if input_type == "SMILES":
        s = (smiles_text or "").strip()
        if not s:
            raise ValueError("SMILES is empty.")
        ligands_raw.append({"name": output_name or "ligand", "smiles": s})

    elif input_type == "SMI_FILE":
        if not uploaded_bytes:
            raise ValueError("No .smi file uploaded.")
        for smi, name in parse_smi_lines(uploaded_bytes.decode("utf-8", errors="replace")):
            ligands_raw.append({"name": name, "smiles": smi})

    elif input_type == "FILE":
        if not uploaded_bytes or not uploaded_name:
            raise ValueError("No ligand file uploaded.")
        ext      = os.path.splitext(uploaded_name)[1].lower()
        tmp_path = out / f"uploaded{ext}"
        tmp_path.write_bytes(uploaded_bytes)
        mol_in = mol_from_file(str(tmp_path))
        try:
            frags = Chem.GetMolFrags(mol_in, asMols=True, sanitizeFrags=False)
            if len(frags) > 1:
                mol_in = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            Chem.SanitizeMol(mol_in)
        except Exception:
            pass
        base_smi = Chem.MolToSmiles(Chem.RemoveHs(mol_in), canonical=True)
        ligands_raw.append({"name": output_name or os.path.splitext(uploaded_name)[0],
                             "smiles": base_smi})
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    # ── MAIN WORKFLOW  (mirrors notebook lines 1241-1355) ───────────────────
    results:         list[dict] = []
    all_micro_rows:  list[dict] = []
    format_warnings: list[str]  = []
    keep_stereo = not enumerate_stereoisomers

    for ligand in ligands_raw:
        base_name   = ligand["name"]
        stereo_rows = enumerate_stereo(ligand["smiles"], keep_original=keep_stereo)

        for _si, (raw_smiles, stereo_tag) in enumerate(stereo_rows, 1):
            pretty = base_name if stereo_tag is None else f"{base_name}_{stereo_tag}"
            print(f"\n{SEP}\n🧪  {pretty}\n{SEP}")

            can_smi, status = standardize_smiles(raw_smiles)
            if can_smi is None:
                print(f"❌  {status}")
                format_warnings.append(f"Standardization failed for {pretty}: {status}")
                continue
            print(f"    SMILES (std): {can_smi}")

            # PubChem lookup
            print("    🔍  PubChem lookup … ", end="", flush=True)
            pc: dict = {}
            if use_pubchem and _REQUESTS_OK:
                try:
                    pc = pubchem_lookup(can_smi)
                    if pc["available"]:
                        print(f"✅  CID={pc['cid']}  pKa={pc['pka_values']}"
                              f"  flags={[k for k, v in pc['flags'].items() if v]}")
                    else:
                        print(f"—  {pc.get('error', 'no data')}")
                except Exception as e:
                    print(f"—  error: {e}")
            else:
                print("—  disabled")

            # Generate ranked microstates
            top, ambig, all_m, tr_flag, tr_motifs, ml_preds = generate_ranked_microstates(
                can_smi,
                target_ph      = target_pH,
                ph_window      = ph_window,
                max_tautomers  = max_tautomers,
                top_n          = top_n_microstates,
                pubchem_result = pc,
            )

            if not top:
                print("❌  No valid microstates generated.")
                format_warnings.append(f"No valid microstates for {pretty}")
                continue

            t = top[0]
            print(f"\n🔬  Microstates generated : {len(all_m)}")
            print(f"⚠️   Ambiguous top state   : {'YES' if ambig else 'NO'}")
            if tr_flag:
                print(f"🔄  Tautomer-rich motifs  : {', '.join(tr_motifs)}")
            print(f"\n🏆  Rank-1")
            for label, val in [
                ("Score",       f"{t['selection_score']:.3f}"),
                ("SMILES",      t["microstate_smiles"]),
                (f"Charge @ pH {target_pH}", f"{t['net_charge']:+d}"),
                ("Zwitterion",  "YES" if t["is_zwitterion_strict"]                else "NO"),
                ("Amide kept",  "YES" if t["flag_amide_preserved"]                else "NO"),
                ("Imidic acid", "YES ⚠️" if t["flag_imidic_acid_penalty"]         else "NO"),
                ("[N-]C=O",     "YES ⚠️" if t["flag_amide_n_deprotonation_penalty"] else "NO"),
                ("Aromaticity", "LOST ⚠️" if t.get("flag_aromaticity_lost")        else "OK"),
                ("pKa source",  t["pKa_source"]),
                ("Backend",     f"{t['decision_backend']}  ({t['decision_mode']})"),
            ]:
                print(f"    {label:<14}: {val}")

            # Microstate CSV
            micro_csv = str(out / f"{pretty}_microstates.csv")
            try:
                import pandas as pd
                pd.DataFrame(
                    [{k: v for k, v in r.items() if k != "charged_atom_rows"} for r in top]
                ).to_csv(micro_csv, index=False)
                print(f"\n💾  {Path(micro_csv).name}")
            except Exception as e:
                print(f"CSV write failed: {e}")
                micro_csv = None

            # 3D structures for top-k microstates
            alt3d: list[tuple] = []
            for row in top[:max(1, write_alt_3d_for_top_k)]:
                rk = row["microstate_rank"]
                bp = str(out / f"{pretty}_micro{rk}_min")
                try:
                    m3d   = build_minimized_3d(row["microstate_smiles"])
                    files = save_molecule_files(m3d, bp, output_formats)
                    for w in files.pop("warnings", []):
                        if w not in format_warnings:
                            format_warnings.append(w)
                    save_2d_image(row["microstate_smiles"], f"{bp}_2D.png")
                    alt3d.append((rk, files.get("pdb"), files.get("sdf"), files))
                except Exception as e:
                    print(f"⚠️  3D failed for rank {rk}: {e}")

            sel_pdb  = alt3d[0][1] if alt3d else None
            sel_sdf  = alt3d[0][2] if alt3d else None
            sel_mol2 = alt3d[0][3].get("mol2") if alt3d else None
            if sel_pdb:
                print(f"💾  {Path(sel_pdb).name}, {Path(sel_sdf).name if sel_sdf else 'no sdf'}")

            # Result dict — exact keys from notebook (plus flag_aromaticity_lost)
            results.append({
                "name":                       pretty,
                "base_smiles":                can_smi,
                "stereo":                     stereo_tag,
                "selected_microstate_smiles": t["microstate_smiles"],
                "selected_tautomer_smiles":   t["tautomer_smiles"],
                "pKa_source":                 t["pKa_source"],
                "decision_backend":           t["decision_backend"],
                "decision_mode":              t["decision_mode"],
                "pubchem_pka_values":         str(pc.get("pka_values", [])),
                "pubchem_confidence":         pc.get("confidence", "n/a"),
                "pubchem_cid":                pc.get("cid"),
                "formal_charge":              t["net_charge"],
                "is_zwitterion":              t["is_zwitterion_strict"],
                "charged_atoms":              t["charged_atoms"],
                "ambiguous_top_assignment":   ambig,
                "flag_tautomer_rich":         tr_flag,
                "flag_tautomer_motifs":       tr_motifs,
                "flag_amide_preserved":       t["flag_amide_preserved"],
                "flag_imidic_acid_penalty":   t["flag_imidic_acid_penalty"],
                "flag_amide_n_deprotonation": t["flag_amide_n_deprotonation_penalty"],
                "flag_aromaticity_lost":      t.get("flag_aromaticity_lost", False),
                "flag_borderline_pka":        t["flag_borderline_pka"],
                "flag_unipka_used":           t["flag_unipka_used"],
                "microstate_csv":             micro_csv,
                "minimized_pdb":              sel_pdb,
                "minimized_sdf":              sel_sdf,
                # Streamlit extras
                "minimized_mol2":             sel_mol2,
                "selection_score":            t["selection_score"],
                "n_all_microstates":          len(all_m),
                "top_microstates":            [{k: v for k, v in r.items()
                                                if k != "charged_atom_rows"} for r in top],
                "alt3d":                      [{"rank": r[0], "pdb": r[1], "sdf": r[2],
                                                "files": r[3],
                                                "smiles": top[i]["microstate_smiles"]}
                                               for i, r in enumerate(alt3d)],
            })

            for row in top:
                all_micro_rows.append(
                    {**{k: v for k, v in row.items() if k != "charged_atom_rows"},
                     "name": pretty, "target_pH": target_pH, "pubchem_cid": pc.get("cid")}
                )

    # ── Summary (mirrors notebook lines 1358-1408) ───────────────────────────
    print(f"\n{SEP}\n📊  SUMMARY  |  pH={target_pH}  |  backend={_PKA_BACKEND}\n{SEP}")
    for r in results:
        print(f"\n▶  {r['name']}")
        for k, v in [
            ("Selected SMILES",             r["selected_microstate_smiles"]),
            (f"Charge @ pH {target_pH}",    f"{r['formal_charge']:+d}"),
            ("Charged atoms",               r["charged_atoms"]),
            ("Zwitterion",                  "YES" if r["is_zwitterion"]               else "NO"),
            ("pKa source",                  r["pKa_source"]),
            ("PubChem pKa",                 r["pubchem_pka_values"]),
            ("Amide preserved",             "YES" if r["flag_amide_preserved"]        else "NO"),
            ("Imidic acid flag",            "YES ⚠️" if r["flag_imidic_acid_penalty"] else "NO"),
            ("[N-]C=O flag",                "YES ⚠️" if r["flag_amide_n_deprotonation"] else "NO"),
            ("Aromaticity lost",            "YES ⚠️" if r.get("flag_aromaticity_lost") else "NO"),
            ("Ambiguous",                   "YES" if r["ambiguous_top_assignment"]    else "NO"),
        ]:
            print(f"   {k:<28}: {v}")

    summary_lines = [
        SEP,
        f"pKaNET Cloud — SUMMARY  |  pH={target_pH}  |  pKa backend={_PKA_BACKEND}",
        SEP,
        f"Structures: {len(results)}  |  pH window: ±{ph_window/2:.1f}  "
        f"|  max tautomers: {max_tautomers}  |  top microstates: {top_n_microstates}",
        "",
    ]
    for r in results:
        summary_lines += [
            f"▶  {r['name']}",
            f"   Selected SMILES   : {r['selected_microstate_smiles']}",
            f"   Charge @ pH {target_pH} : {r['formal_charge']:+d}",
            f"   Charged atoms     : {r['charged_atoms']}",
            f"   Zwitterion        : {'YES' if r['is_zwitterion'] else 'NO'}",
            f"   Ambiguous         : {'YES' if r['ambiguous_top_assignment'] else 'NO'}",
            f"   pKa source        : {r['pKa_source']}",
            f"   PubChem pKa       : {r['pubchem_pka_values']} (conf={r['pubchem_confidence']})",
            f"   Amide preserved   : {'YES' if r['flag_amide_preserved'] else 'NO'}",
            f"   Imidic acid       : {'YES ⚠️' if r['flag_imidic_acid_penalty'] else 'NO'}",
            f"   [N-]C=O           : {'YES ⚠️' if r['flag_amide_n_deprotonation'] else 'NO'}",
            f"   Aromaticity lost  : {'YES ⚠️' if r.get('flag_aromaticity_lost') else 'NO'}",
            "",
        ]
    summary_text = "\n".join(summary_lines)
    (out / "summary.txt").write_text(summary_text + "\n")

    if input_type == "SMI_FILE" and results:
        log_lines = [
            "# pKaNET Cloud — Processing Log",
            f"# pH={target_pH}  backend={_PKA_BACKEND}",
            "# Name | pH-SMILES | Charge | Zwitterion | Ambiguous | pKa_source | PubChem_pKa",
            "",
        ]
        for r in results:
            log_lines.append(
                f"{r['name']}\t{r['selected_microstate_smiles']}\t{r['formal_charge']:+d}\t"
                f"{'Yes' if r['is_zwitterion'] else 'No'}\t"
                f"{'Yes' if r['ambiguous_top_assignment'] else 'No'}\t"
                f"{r['pKa_source']}\t{r['pubchem_pka_values']}"
            )
        (out / "processing.log").write_text("\n".join(log_lines) + "\n")

    return {"results": results, "summary_text": summary_text,
            "out_dir": str(out), "format_warnings": format_warnings,
            "pka_backend": _PKA_BACKEND}

# ─────────────────────────────────────────────────────────────────────────────
# ZIP helpers
# ─────────────────────────────────────────────────────────────────────────────

def zip_minimized_structures(out_dir: str, zip_path: str, selected_formats: list[str]) -> str:
    out = Path(out_dir)
    zp  = Path(zip_path)
    fmts = [f.lower() for f in selected_formats]
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.glob("*_min.*"):
            s = p.suffix.lower()
            if (s == ".pdb" and "pdb" in fmts) or (s == ".mol2" and "mol2" in fmts):
                z.write(p, arcname=p.name)
    return str(zp)


def zip_all_outputs(out_dir: str, zip_path: str) -> str:
    out = Path(out_dir)
    zp  = Path(zip_path)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(out))
    return str(zp)
