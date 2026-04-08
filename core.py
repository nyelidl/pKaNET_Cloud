# core.py  —  pKaNET Cloud v4
# Tautomer-aware microstate ranking + pH-adjusted SMILES + minimized 3D
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TAUTOMER_PLAUSIBILITY_CUTOFF = 3.0   # max score gap to keep a tautomer
AMBIGUITY_SCORE_GAP          = 0.5   # rank-1 vs rank-2 gap to flag ambiguity
BORDERLINE_PKA_WINDOW        = 1.0   # |pH – pKa| ≤ this → borderline flag
PUBCHEM_RATE_LIMIT_S         = 0.25  # seconds between PubChem requests
PUBCHEM_CACHE_FILE           = "/tmp/pkanet_pubchem_cache.json"

# ─────────────────────────────────────────────────────────────────────────────
# Optional dependency probes
# ─────────────────────────────────────────────────────────────────────────────
try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _requests = None
    _REQUESTS_OK = False

try:
    from dimorphite_dl import protonate_smiles as _dimorphite_fn
    _DIMORPHITE_OK = True
except ImportError:
    _dimorphite_fn = None
    _DIMORPHITE_OK = False

_PKASOLVER_OK = False
_PROPKA_OK    = False
_UNIPKA_OK    = False
_PKA_BACKEND  = "pkapredict"   # default: pKaPredict ML model

try:
    from pkasolver.query import QueryModel as _PkaSolverModel
    _PKASOLVER_OK = True
    _PKA_BACKEND  = "pkasolver"
except ImportError:
    pass

if not _PKASOLVER_OK:
    try:
        import propka.run as _propka_run  # noqa: F401
        _PROPKA_OK   = True
        _PKA_BACKEND = "propka"
    except ImportError:
        pass

if not _PKASOLVER_OK and not _PROPKA_OK:
    if shutil.which("unipka"):
        _UNIPKA_OK   = True
        _PKA_BACKEND = "unipka_cli"

# ─────────────────────────────────────────────────────────────────────────────
# Open Babel helper
# ─────────────────────────────────────────────────────────────────────────────
_OBABEL_AVAILABLE: Optional[bool] = None

def check_obabel() -> bool:
    global _OBABEL_AVAILABLE
    if _OBABEL_AVAILABLE is None:
        _OBABEL_AVAILABLE = shutil.which("obabel") is not None
    return _OBABEL_AVAILABLE


def convert_pdb_to_mol2_obabel(pdb_path: str, mol2_path: str) -> bool:
    if not check_obabel():
        return False
    try:
        result = subprocess.run(
            ["obabel", pdb_path, "-O", mol2_path],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and Path(mol2_path).exists()
    except Exception as e:
        print(f"Open Babel error: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# STAGE A  ·  SMILES standardization
# ─────────────────────────────────────────────────────────────────────────────

def standardize_smiles(smiles: str) -> Tuple[Optional[str], str]:
    """Validate, normalize, and canonicalize.  Returns (canonical, status)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, f"RDKit cannot parse: {smiles[:80]}"
    mol = Chem.RemoveHs(mol, implicitOnly=True)
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    try:
        mol = rdMolStandardize.Normalizer().normalize(mol)
    except Exception:
        pass
    can = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return can, "OK"


def canonicalize(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def smiles_to_inchikey(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# STAGE B  ·  PubChem experimental pKa retrieval
# ─────────────────────────────────────────────────────────────────────────────
_PUBCHEM_CACHE: Dict[str, Any] = {}

def _load_pubchem_cache() -> None:
    global _PUBCHEM_CACHE
    try:
        if Path(PUBCHEM_CACHE_FILE).exists():
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


def _pubchem_get(url: str, timeout: int = 12) -> Optional[dict]:
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


def _flatten_pubchem_section(section: dict, target_heading: str) -> List[str]:
    results: List[str] = []
    if target_heading.lower() in section.get("TOCHeading", "").lower():
        for info in section.get("Information", []):
            for swm in info.get("Value", {}).get("StringWithMarkup", []):
                s = swm.get("String", "").strip()
                if s:
                    results.append(s)
    for sub in section.get("Section", []):
        results.extend(_flatten_pubchem_section(sub, target_heading))
    return results


def pubchem_lookup(smiles: str) -> dict:
    """Retrieve PubChem experimental pKa evidence. Returns dict with availability/values/flags."""
    result = dict(available=False, cid=None, inchikey=None,
                  pka_values=[], source_texts=[], flags={}, confidence="low", error=None)
    if not _REQUESTS_OK:
        result["error"] = "requests not installed"
        return result

    ik = smiles_to_inchikey(smiles)
    if ik is None:
        result["error"] = "InChIKey computation failed."
        return result
    result["inchikey"] = ik

    # CID lookup
    cid_key = f"cid:{ik}"
    if cid_key in _PUBCHEM_CACHE:
        cid = _PUBCHEM_CACHE[cid_key]
    else:
        data = _pubchem_get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{ik}/cids/JSON"
        )
        cid = None
        if data:
            try:
                cid = int(data["IdentifierList"]["CID"][0])
            except Exception:
                pass
        _PUBCHEM_CACHE[cid_key] = cid
        _save_pubchem_cache()

    if cid is None:
        result["error"] = "CID not found on PubChem."
        return result
    result["cid"] = cid

    # Dissociation constants
    diss_key = f"diss:{cid}"
    if diss_key in _PUBCHEM_CACHE:
        texts = _PUBCHEM_CACHE[diss_key]
    else:
        data = _pubchem_get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
            "?heading=Dissociation+Constants"
        )
        texts: List[str] = []
        if data:
            try:
                for sec in data.get("Record", {}).get("Section", []):
                    texts.extend(_flatten_pubchem_section(sec, "Dissociation"))
            except Exception:
                pass
        _PUBCHEM_CACHE[diss_key] = texts
        _save_pubchem_cache()

    if not texts:
        result["error"] = "No dissociation constant data on PubChem."
        return result

    # Parse pKa values
    full_text = " ".join(texts).lower()
    found: List[float] = []
    src: List[str] = []
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

    dedup: List[float] = []
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
        "conflicting_values":    conflicting,
        "vague_or_approximate":  vague,
        "confidence":            confidence,
    }
    result.update(
        available    = bool(dedup),
        pka_values   = dedup,
        source_texts = src,
        flags        = flags,
        confidence   = confidence,
    )
    return result

# ─────────────────────────────────────────────────────────────────────────────
# STAGE C  ·  ML pKa backends  (pkasolver / propka / unipka / pKaPredict)
# ─────────────────────────────────────────────────────────────────────────────

def _unipka_via_pkasolver(smiles: str) -> List[dict]:
    try:
        from pkasolver.query import QueryModel
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        df = QueryModel().predict_pka(mol)
        return [
            {"pka": float(row.get("pKa", row.get("pka", 0))),
             "site_type": str(row.get("type", "?")),
             "site_label": str(row.get("atom_idx", "?")),
             "source": "pkasolver", "confidence": "ml_gnn"}
            for _, row in df.iterrows()
        ]
    except Exception as e:
        print(f"pkasolver failed: {e}")
        return []


def _unipka_via_propka(smiles: str) -> List[dict]:
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
                    results.append({
                        "pka": float(pv),
                        "site_label": str(getattr(grp, "atom_name", "?")),
                        "site_type": str(getattr(grp, "type", "?")),
                        "source": "propka", "confidence": "semi_empirical",
                    })
        finally:
            try:
                os.unlink(tmppath)
            except Exception:
                pass
        return results
    except Exception as e:
        print(f"propka failed: {e}")
        return []


def _unipka_via_cli(smiles: str) -> List[dict]:
    try:
        r = subprocess.run(
            ["unipka", "--smiles", smiles, "--json"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return []
        data = json.loads(r.stdout)
        return [
            {"pka": entry.get("pka"), "site_label": entry.get("site", "?"),
             "site_type": entry.get("type", "?"), "source": "unipka_cli", "confidence": "ml"}
            for entry in data.get("microstates", [])
        ]
    except Exception as e:
        print(f"unipka CLI failed: {e}")
        return []


def unipka_predict(smiles: str) -> List[dict]:
    """Try ML pKa backends in priority order."""
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

# ─────────────────────────────────────────────────────────────────────────────
# IUPAC + pKaPredict  (legacy fallback)
# ─────────────────────────────────────────────────────────────────────────────
_IUPAC_DF          = None
_PKA_MAP_ALL: Optional[dict] = None
_IUPAC_LOADED      = False
_PKANET_MODEL      = None
_DESCRIPTOR_NAMES  = None


def load_iupac_dataset() -> None:
    global _IUPAC_DF, _PKA_MAP_ALL, _IUPAC_LOADED
    if _IUPAC_LOADED:
        return
    _IUPAC_LOADED = True
    try:
        import pandas as pd
        IUPAC_CSV_URL = (
            "https://raw.githubusercontent.com/IUPAC/Dissociation-Constants/"
            "main/iupac_high-confidence_v2_3.csv"
        )
        print("Loading IUPAC pKa dataset…")
        iupac_df = pd.read_csv(IUPAC_CSV_URL)
        cols = list(iupac_df.columns)
        lower_map = {c.lower(): c for c in cols}

        smiles_col = next(
            (c for w in ["SMILES", "smiles"] for c in [w, lower_map.get(w.lower())] if c and c in cols), None
        )
        pka_col = next(
            (c for w in ["pka_value", "pKa", "pka", "value"] for c in [w, lower_map.get(w.lower())] if c and c in cols), None
        )
        if not smiles_col or not pka_col:
            print("IUPAC: cannot find SMILES/pKa columns")
            return

        def _can(smi):
            m = Chem.MolFromSmiles(str(smi).strip())
            return Chem.MolToSmiles(m, canonical=True) if m else None

        iupac_df["_cansmi"] = iupac_df[smiles_col].apply(_can)
        pka_map: Dict[str, List[float]] = defaultdict(list)
        for csmi, pka in zip(iupac_df["_cansmi"], iupac_df[pka_col]):
            if csmi is None:
                continue
            try:
                pka_map[csmi].append(float(pka))
            except Exception:
                pass
        _IUPAC_DF = iupac_df
        _PKA_MAP_ALL = pka_map
        print(f"IUPAC loaded: {len(pka_map):,} molecules")
    except Exception as e:
        print(f"IUPAC load failed: {e}")


def lookup_pka_iupac(query_smiles: str) -> Optional[Dict[str, Any]]:
    if _PKA_MAP_ALL is None:
        return None
    mol = Chem.MolFromSmiles(query_smiles)
    if mol is None:
        return None
    canonical_smi = Chem.MolToSmiles(mol, canonical=True)
    vals = _PKA_MAP_ALL.get(canonical_smi, [])
    if not vals:
        return None
    vals = sorted(vals)
    return {
        "pka_median": float(np.median(vals)),
        "n": len(vals),
        "all": vals,
    }


def get_pkapredict_model():
    global _PKANET_MODEL, _DESCRIPTOR_NAMES
    if _PKANET_MODEL is None:
        from pkapredict import load_model
        _PKANET_MODEL = load_model()
        if hasattr(_PKANET_MODEL, "feature_name_"):
            _DESCRIPTOR_NAMES = _PKANET_MODEL.feature_name_
        else:
            from rdkit.Chem import Descriptors
            all_desc = [d[0] for d in Descriptors._descList]
            _DESCRIPTOR_NAMES = all_desc[:_PKANET_MODEL.n_features_]
    return _PKANET_MODEL, _DESCRIPTOR_NAMES


def predict_pka_pkanet(smiles: str) -> float:
    from pkapredict import predict_pKa
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Cannot parse SMILES for pKa prediction.")
    model, descriptor_names = get_pkapredict_model()
    pka_value = predict_pKa(smiles, model, descriptor_names)
    if isinstance(pka_value, (list, tuple)):
        pka_value = pka_value[0]
    elif hasattr(pka_value, "__iter__") and not isinstance(pka_value, str):
        pka_value = next(iter(pka_value))
    return float(pka_value)


def get_pka_scalar(smiles: str, use_iupac: bool = True) -> Tuple[Optional[float], str]:
    """Return (pKa, source_string) using best available method."""
    if use_iupac and _PKA_MAP_ALL is not None:
        stats = lookup_pka_iupac(smiles)
        if stats is not None:
            return stats["pka_median"], f"IUPAC (n={stats['n']})"
    try:
        return predict_pka_pkanet(smiles), "pKaPredict (ML)"
    except Exception as e:
        print(f"pKaPredict failed: {e}")
    return None, "heuristic"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE D  ·  Dimorphite-DL protonation enumerator
# ─────────────────────────────────────────────────────────────────────────────

def dimorphite_enumerate(
    smiles: str,
    ph_min: float,
    ph_max: float,
    precision: float = 1.0,
    max_variants: int = 128,
) -> List[str]:
    """Enumerate protonation states; always includes input SMILES."""
    if not _DIMORPHITE_OK or _dimorphite_fn is None:
        return [smiles]

    kwarg_variants = [
        {"ph_min": ph_min, "ph_max": ph_max, "precision":     precision, "max_variants": max_variants},
        {"min_ph": ph_min, "max_ph": ph_max, "pka_precision": precision, "max_variants": max_variants},
        {"ph_min": ph_min, "ph_max": ph_max, "precision":     precision},
        {"min_ph": ph_min, "max_ph": ph_max, "pka_precision": precision},
    ]
    raw: List[str] = []
    for kwargs in kwarg_variants:
        try:
            r = _dimorphite_fn(smiles, **kwargs)
            raw = [r] if isinstance(r, str) else list(r or [])
            if raw:
                break
        except TypeError:
            pass

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
            r = _dimorphite_fn(smiles, **kw)
            raw = [r] if isinstance(r, str) else list(r or [])
        except Exception as e:
            print(f"dimorphite-dl failed: {e}")

    seen: set = set()
    result: List[str] = []
    seed = canonicalize(smiles)
    if seed:
        seen.add(seed); result.append(seed)
    for smi in raw:
        c = canonicalize(smi)
        if c and c not in seen:
            seen.add(c); result.append(c)
    return result or [smiles]

# ─────────────────────────────────────────────────────────────────────────────
# STAGE E  ·  Henderson–Hasselbalch scoring
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
        if expected_neg and actual_charge < 0:
            return  min(1.2, dpH * 0.45)
        elif expected_neg:
            return -min(1.0, dpH * 0.35)
        elif actual_charge >= 0:
            return  0.1
        else:
            return -min(1.2, dpH * 0.40)
    else:
        expected_pos = f_charged > 0.5
        if expected_pos and actual_charge > 0:
            return  min(1.2, dpH * 0.45)
        elif expected_pos:
            return -min(1.0, dpH * 0.35)
        elif actual_charge <= 0:
            return  0.1
        else:
            return -min(1.2, dpH * 0.40)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE F  ·  Ionizable site table + tautomer plausibility
# ─────────────────────────────────────────────────────────────────────────────

_IONIZABLE_SITE_DEF = [
    ("sulfonic_acid",      "[SX4](=O)(=O)[OX2H1]",                              1.0,  "acid"),
    ("carboxylic_acid",    "[CX3](=O)[OX2H1]",                                  4.5,  "acid"),
    ("tetrazole",          "c1nn[nH]n1",                                         4.9,  "acid"),
    ("imidazole",          "c1cn[nH]c1",                                         6.0,  "acid"),
    ("benzimidazole",      "c1ccc2[nH]cnc2c1",                                  5.5,  "acid"),
    ("phosphonate",        "[PX4](=O)([OX2H1])[OX2H1,OX1-]",                   6.5,  "acid"),
    ("sulfonamide_NH",     "[SX4](=O)(=O)[NX3;H1]",                            10.1,  "acid"),
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

_IONIZABLE_SITES_COMPILED: List[tuple] = []
for _lbl, _sma, _pka_val, _typ in _IONIZABLE_SITE_DEF:
    _pat = Chem.MolFromSmarts(_sma)
    if _pat is not None:
        _IONIZABLE_SITES_COMPILED.append((_lbl, _pat, _pka_val, _typ))


def find_ionizable_sites(mol: Chem.Mol) -> List[dict]:
    sites = []
    seen_k: set = set()
    for lbl, pat, pka_val, stype in _IONIZABLE_SITES_COMPILED:
        for match in mol.GetSubstructMatches(pat):
            k = frozenset(match)
            if k in seen_k:
                continue
            seen_k.add(k)
            sites.append(dict(label=lbl, atom_indices=list(match),
                               heuristic_pka=pka_val, site_type=stype))
    return sites


_BONUS_DEF = [
    ("amide",            +2.5, "[CX3](=O)[NX3;H1,H2]"),
    ("lactam",           +2.5, "[C;R](=O)[N;R]"),
    ("acylhydrazone_NH", +2.0, "[CX3](=O)[NX3;H1][NX2]=[CX3]"),
    ("hydrazide_NH",     +2.0, "[CX3](=O)[NX3;H1][NX3;H2]"),
    ("urea_NH",          +1.5, "[NX3;H1][CX3](=O)[NX3;H1,H2]"),
    ("aromatic_ring",    +0.3, "c1ccccc1"),
]
_PENALTY_DEF = [
    ("imidic_acid_open", -4.0, "[CX3;!R](=[NX2])[OX2H1]"),
    ("lactim_ring",      -4.0, "[C;R](=[NX2])[OX2H1]"),
    ("iminol_general",   -3.5, "[NX2]=[CX3][OX2H1]"),
    ("amide_N_deproton", -5.0, "[$([NX3-]C=O),$([NX3-]c=O)]"),
    ("enol_simple",      -1.2, "[CX3](=[CX3])[OX2H1]"),
]
_CHEM_RULES: List[tuple] = []
for _lbl, _wt, _sma in _BONUS_DEF + _PENALTY_DEF:
    _pat = Chem.MolFromSmarts(_sma)
    if _pat is not None:
        _CHEM_RULES.append((_lbl, _wt, _pat))

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


def score_tautomer_plausibility(smiles: str) -> Tuple[float, dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -999.0, {}
    bd: Dict[str, float] = {}
    total = 0.0
    for lbl, wt, pat in _CHEM_RULES:
        n = len(mol.GetSubstructMatches(pat))
        if n:
            c = wt * n
            bd[lbl] = round(c, 3)
            total  += c
    bd["_total"] = round(total, 3)
    return total, bd


def is_tautomer_rich(mol: Chem.Mol) -> Tuple[bool, List[str]]:
    hits = [l for l, p in _TAUTOMER_RICH_COMPILED if mol.HasSubstructMatch(p)]
    return bool(hits), hits


def enumerate_and_filter_tautomers(
    smiles: str,
    max_states: int = 8,
    cutoff: float = TAUTOMER_PLAUSIBILITY_CUTOFF,
) -> Tuple[List[dict], List[dict], bool, List[str]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles[:60]}")

    tr_flag, tr_motifs = is_tautomer_rich(mol)
    enum   = rdMolStandardize.TautomerEnumerator()
    seen:  set = set()
    scored: List[dict] = []

    for tmol in enum.Enumerate(mol):
        smi = Chem.MolToSmiles(tmol, isomericSmiles=True, canonical=True)
        if smi in seen:
            continue
        seen.add(smi)
        sc, bd = score_tautomer_plausibility(smi)
        scored.append({"smiles": smi, "score": sc, "breakdown": bd})

    if not scored:
        smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        sc, bd = score_tautomer_plausibility(smi)
        scored = [{"smiles": smi, "score": sc, "breakdown": bd}]

    scored = sorted(scored[:max_states], key=lambda x: -x["score"])
    best       = scored[0]["score"]
    eff_cutoff = cutoff * (2.0 if tr_flag else 1.0)
    kept       = [t for t in scored if t["score"] >= best - eff_cutoff]
    discarded  = [t for t in scored if t["score"] <  best - eff_cutoff]
    return kept or [scored[0]], discarded, tr_flag, tr_motifs

# ─────────────────────────────────────────────────────────────────────────────
# STAGE G  ·  Charge profile
# ─────────────────────────────────────────────────────────────────────────────

def get_charge_profile(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles[:60]}")
    net = n_pos = n_neg = 0
    rows = []
    for atom in mol.GetAtoms():
        fc = int(atom.GetFormalCharge())
        net  += fc
        n_pos += fc > 0
        n_neg += fc < 0
        if fc != 0:
            rows.append({"atom_idx": atom.GetIdx(),
                          "symbol": atom.GetSymbol(),
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
    return ", ".join(
        f"{r['symbol']}{r['atom_idx']}({r['formal_charge']:+d})" for r in rows
    )

# ─────────────────────────────────────────────────────────────────────────────
# STAGE H  ·  Microstate scoring + ranking
# ─────────────────────────────────────────────────────────────────────────────

def _best_pka_for_site(
    site: dict,
    ml_predictions: List[dict],
    pubchem_result: dict,
) -> Tuple[float, str]:
    stype = site["site_type"]
    for mp in ml_predictions:
        if mp.get("pka") is not None and mp.get("site_type", "").lower() == stype:
            return float(mp["pka"]), mp.get("source", "ml")
    if pubchem_result.get("available") and \
            pubchem_result.get("confidence") in ("high", "medium"):
        vals = pubchem_result.get("pka_values", [])
        if vals:
            best_pc = min(vals, key=lambda v: abs(v - site["heuristic_pka"]))
            return best_pc, "pubchem"
    return site["heuristic_pka"], "heuristic"


def score_microstate_full(
    microstate_smiles: str,
    tautomer_smiles:   str,
    taut_plausibility: float,
    taut_breakdown:    dict,
    ion_sites:         List[dict],
    ml_predictions:    List[dict],
    pubchem_result:    dict,
    target_ph:         float,
) -> Tuple[float, dict, dict, bool]:
    mol = Chem.MolFromSmiles(microstate_smiles)
    if mol is None:
        return -1e9, {}, {}, False

    cp  = get_charge_profile(microstate_smiles)
    net = cp["net_charge"]
    n_pos, n_neg = cp["n_pos_atoms"], cp["n_neg_atoms"]
    fc_map = {a.GetIdx(): a.GetFormalCharge() for a in mol.GetAtoms()}

    # Layer 1 – safety: amide-N deprotonation
    pat_amide_neg = Chem.MolFromSmarts("[$([NX3-]C=O),$([NX3-]c=O)]")
    n_amide_neg   = len(mol.GetSubstructMatches(pat_amide_neg)) if pat_amide_neg else 0
    s_amide_n_dep = -5.0 * n_amide_neg

    # Layer 2 – tautomer plausibility
    s_tautomer = 0.65 * taut_plausibility

    # Layer 3 – pH consistency (HH)
    borderline = False
    ph_bd: Dict[str, float] = {}
    s_ph = 0.0
    for site in ion_sites:
        pka_val, pka_src = _best_pka_for_site(site, ml_predictions, pubchem_result)
        if abs(target_ph - pka_val) <= BORDERLINE_PKA_WINDOW:
            borderline = True
        site_charge = sum(fc_map.get(i, 0) for i in site["atom_indices"])
        contrib     = hh_ph_match_score(pka_val, target_ph, site["site_type"], site_charge)
        ph_bd[f"pH_{site['label']}[{pka_src}]"] = round(contrib, 3)
        s_ph += contrib

    # Layer 4 – PubChem evidence bonus
    s_pubchem_bonus = 0.0
    if pubchem_result.get("available"):
        pc_weight = {"high": 1.0, "medium": 0.6, "low": 0.2}.get(
            pubchem_result.get("confidence", "low"), 0.2)
        for pka_val in pubchem_result["pka_values"]:
            exp = -1 if hh_fraction_charged(pka_val, target_ph, "acid") > 0.5 else 0
            s_pubchem_bonus += 0.25 * pc_weight if net == exp else -0.15 * pc_weight
        s_pubchem_bonus = max(-0.4, min(0.5, s_pubchem_bonus))

    # Layer 5 – charge-structure reasonableness
    has_acid_site = any(s["site_type"] == "acid" and (target_ph - s["heuristic_pka"]) > 1.0 for s in ion_sites)
    has_base_site = any(s["site_type"] == "base" and (s["heuristic_pka"] - target_ph) > 1.0 for s in ion_sites)

    if cp["is_zwitterion_strict"]:
        s_zwit = 0.8 if (has_acid_site and has_base_site) else -0.6
    else:
        s_zwit = -0.4 if (has_acid_site and has_base_site and net == 0 and n_pos == 0) else 0.0

    strong_acid = [s for s in ion_sites if s["site_type"] == "acid" and (target_ph - s["heuristic_pka"]) > 2.0]
    strong_base = [s for s in ion_sites if s["site_type"] == "base" and (s["heuristic_pka"] - target_ph) > 2.0]
    s_improbable = 0.0
    if strong_acid and net >= 0 and n_neg == 0:
        s_improbable -= 0.5 * len(strong_acid)
    if strong_base and net <= 0 and n_pos == 0:
        s_improbable -= 0.5 * len(strong_base)
    s_multi = -0.12 * max(0, n_pos + n_neg - 2)

    total = s_amide_n_dep + s_tautomer + s_ph + s_pubchem_bonus + s_zwit + s_improbable + s_multi

    flag_amide   = any(taut_breakdown.get(k, 0) > 0 for k in ["amide","lactam","acylhydrazone_NH","hydrazide_NH"])
    flag_imidic  = any(taut_breakdown.get(k, 0) < 0 for k in ["imidic_acid_open","lactim_ring","iminol_general"])
    flag_lactim  = taut_breakdown.get("lactim_ring", 0) < 0
    has_ml       = bool(ml_predictions) or pubchem_result.get("available", False)
    decision_backend = (ml_predictions[0].get("source", "ml") if ml_predictions else
                        ("pubchem" if pubchem_result.get("available") else "heuristic"))

    cp.update(
        flag_amide_preserved               = flag_amide,
        flag_imidic_acid_penalty           = flag_imidic,
        flag_lactim_penalty                = flag_lactim,
        flag_amide_n_deprotonation_penalty = n_amide_neg > 0,
        decision_backend                   = decision_backend,
    )
    bd_full = {
        "s_amide_n_deproton":         round(s_amide_n_dep,   3),
        "s_tautomer_plausibility":    round(s_tautomer,      3),
        "s_ph_consistency":           round(s_ph,            3),
        "s_pubchem_evidence":         round(s_pubchem_bonus, 3),
        "s_zwitterion_consistency":   round(s_zwit,          3),
        "s_improbable_neutral":       round(s_improbable,    3),
        "s_multicharge_penalty":      round(s_multi,         3),
        "total_score":                round(total,           3),
        **ph_bd,
    }
    return total, cp, bd_full, borderline


def generate_ranked_microstates(
    base_smiles:   str,
    target_ph:     float = 7.4,
    ph_window:     float = 1.0,
    max_tautomers: int   = 8,
    top_n:         int   = 5,
    pubchem_result: dict | None = None,
    use_iupac_pka: bool  = True,
) -> Tuple[List[dict], bool, List[dict], bool, List[str], List[dict], Optional[float], str]:
    """
    Full tautomer-aware microstate ranking pipeline.

    Returns
    -------
    top_microstates, ambiguous_flag, all_microstates,
    tautomer_rich_flag, tr_motifs, ml_predictions,
    pka_scalar, pka_source
    """
    if pubchem_result is None:
        pubchem_result = {}

    kept, disc, tr_flag, tr_motifs = enumerate_and_filter_tautomers(
        base_smiles, max_states=max_tautomers, cutoff=TAUTOMER_PLAUSIBILITY_CUTOFF)

    if disc:
        print(f"Discarded {len(disc)} implausible tautomers")

    ml_preds  = unipka_predict(base_smiles)
    base_mol  = Chem.MolFromSmiles(base_smiles)
    ion_sites = find_ionizable_sites(base_mol) if base_mol else []

    # Scalar pKa for display
    pka_scalar, pka_source = get_pka_scalar(base_smiles, use_iupac=use_iupac_pka)
    if pubchem_result.get("available") and pubchem_result.get("pka_values"):
        pka_scalar = float(np.median(pubchem_result["pka_values"]))
        pka_source = f"PubChem (n={len(pubchem_result['pka_values'])})"

    all_micro: List[dict] = []
    seen_smi:  set = set()

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
                )
            except Exception as e:
                print(f"Scoring error ({psmi[:40]}): {e}")
                continue

            pka_src_label = (
                ml_preds[0].get("source", "ml") if ml_preds else
                ("pubchem" if pubchem_result.get("available") else pka_source)
            )
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
                "decision_backend":                  cp.get("decision_backend", "heuristic"),
                "flag_amide_preserved":              cp.get("flag_amide_preserved",              False),
                "flag_imidic_acid_penalty":          cp.get("flag_imidic_acid_penalty",          False),
                "flag_lactim_penalty":               cp.get("flag_lactim_penalty",               False),
                "flag_amide_n_deprotonation_penalty":cp.get("flag_amide_n_deprotonation_penalty",False),
                "flag_borderline_pka":               bl,
                "flag_tautomer_rich":                tr_flag,
                "flag_pubchem_conflicting":          pubchem_result.get("flags",{}).get("conflicting_values", False),
                "flag_pubchem_confidence":           pubchem_result.get("confidence", "n/a"),
                "flag_unipka_used":                  bool(ml_preds),
                "pKa_source":                        pka_src_label,
                **{f"score_{k}": v for k, v in bd.items()},
                **{f"taut_{k}":  v for k, v in taut["breakdown"].items()},
            })

    if not all_micro:
        return [], False, [], tr_flag, tr_motifs, ml_preds, pka_scalar, pka_source

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

    return top, ambiguous, all_micro, tr_flag, tr_motifs, ml_preds, pka_scalar, pka_source

# ─────────────────────────────────────────────────────────────────────────────
# STAGE I  ·  3D structure generation
# ─────────────────────────────────────────────────────────────────────────────

def build_minimized_3d(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles[:60]}")
    mol = Chem.AddHs(mol)

    code = -1
    try:
        try:
            params = AllChem.ETKDGv3()
        except AttributeError:
            params = AllChem.ETKDG()
        params.randomSeed = 42
        code = AllChem.EmbedMolecule(mol, params)
    except Exception:
        code = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=2000)

    if code != 0 or mol.GetNumConformers() == 0:
        code2 = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42, maxAttempts=2000)
        if code2 != 0 or mol.GetNumConformers() == 0:
            raise ValueError("3D embedding failed (no conformer).")

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    return mol

# ─────────────────────────────────────────────────────────────────────────────
# File I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_smi_lines(text: str) -> List[Tuple[str, str]]:
    records = []
    idx = 1
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        smi  = parts[0]
        name = parts[1] if len(parts) > 1 else f"mol_{idx:03d}"
        records.append((smi, name))
        idx += 1
    return records


def generate_RS_variants(base_smiles: str, base_name: str, keep_original: bool = False) -> List[dict]:
    mol = Chem.MolFromSmiles(base_smiles)
    if mol is None:
        return [{"name": base_name, "stereo": None, "base_smiles": base_smiles}]
    if keep_original:
        return [{"name": base_name, "stereo": None,
                  "base_smiles": Chem.MolToSmiles(mol, isomericSmiles=True)}]

    opts   = StereoEnumerationOptions(onlyUnassigned=False)
    isomers = list(EnumerateStereoisomers(mol, options=opts))
    if len(isomers) == 1:
        return [{"name": base_name, "stereo": None,
                  "base_smiles": Chem.MolToSmiles(isomers[0], isomericSmiles=True)}]

    variants = []
    used: set = set()
    for iso in isomers:
        Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
        centers = Chem.FindMolChiralCenters(iso, includeUnassigned=False)
        labs = {lab for _, lab in centers if lab in ("R", "S")}
        label_here = None
        if "R" in labs and "R" not in used:
            label_here = "R"
        elif "S" in labs and "S" not in used:
            label_here = "S"
        if label_here:
            used.add(label_here)
            variants.append({"name": base_name, "stereo": label_here,
                               "base_smiles": Chem.MolToSmiles(iso, isomericSmiles=True)})
        if used == {"R", "S"}:
            break
    return variants or [{"name": base_name, "stereo": None,
                          "base_smiles": Chem.MolToSmiles(isomers[0], isomericSmiles=True)}]


def save_2d_structure_image(smiles: str, output_path: str, size: tuple = (800, 600)) -> bool:
    try:
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=size)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"2D image failed: {e}")
        return False


def save_molecule_files(mol: Chem.Mol, base_path: str, formats: List[str]) -> Dict[str, Any]:
    saved_files: dict = {}
    warnings: List[str] = []
    mol2_via_obabel = False

    # Always save SDF
    try:
        sdf_path = f"{base_path}.sdf"
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        saved_files["sdf"] = sdf_path
    except Exception as e:
        warnings.append(f"Could not save SDF: {e}")

    for fmt in formats:
        fmt_upper = fmt.upper()
        if fmt_upper == "SDF":
            continue
        try:
            if fmt_upper == "PDB":
                fp = f"{base_path}.pdb"
                Chem.MolToPDBFile(mol, fp)
                saved_files["pdb"] = fp

            elif fmt_upper == "MOL2":
                fp = f"{base_path}.mol2"
                if hasattr(Chem, "MolToMol2File"):
                    try:
                        Chem.MolToMol2File(mol, fp)
                        saved_files["mol2"] = fp
                        continue
                    except Exception:
                        pass
                # Fallback: PDB → MOL2 via obabel
                if "pdb" not in saved_files:
                    pdb_fp = f"{base_path}.pdb"
                    Chem.MolToPDBFile(mol, pdb_fp)
                    saved_files["pdb"] = pdb_fp
                if convert_pdb_to_mol2_obabel(saved_files["pdb"], fp):
                    saved_files["mol2"] = fp
                    mol2_via_obabel = True
                else:
                    if not check_obabel():
                        warnings.append("MOL2 unavailable. Install Open Babel (obabel) to enable MOL2 output.")
                    else:
                        warnings.append("MOL2 conversion failed.")
        except Exception as e:
            warnings.append(f"Could not save {fmt_upper}: {e}")

    if mol2_via_obabel:
        warnings.append("ℹ️ MOL2 files generated via Open Babel (converted from PDB)")

    return {"files": saved_files, "warnings": warnings}

# ─────────────────────────────────────────────────────────────────────────────
# Main run_job
# ─────────────────────────────────────────────────────────────────────────────

def run_job(
    *,
    input_type:              str,
    smiles_text:             Optional[str],
    uploaded_bytes:          Optional[bytes],
    uploaded_name:           Optional[str],
    target_pH:               float,
    output_name:             str,
    out_dir:                 str,
    output_formats:          Optional[List[str]] = None,
    enumerate_stereoisomers: bool  = True,
    charge_mode:             str   = "AUTO",     # kept for back-compat; now unused in new pipeline
    use_iupac_pka:           bool  = True,
    use_pubchem:             bool  = True,
    ph_window:               float = 1.0,
    max_tautomers:           int   = 8,
    top_n_microstates:       int   = 5,
    write_alt_3d_for_top_k:  int   = 3,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if output_formats is None or len(output_formats) == 0:
        output_formats = ["PDB"]
    formats_to_save = [fmt.upper() for fmt in output_formats]

    # Load IUPAC dataset
    if use_iupac_pka and not _IUPAC_LOADED:
        try:
            load_iupac_dataset()
        except Exception:
            pass

    # ── Collect raw ligands ──────────────────────────────────────────────────
    ligands_raw: List[dict] = []

    if input_type == "SMILES":
        base_smiles = (smiles_text or "").strip()
        if not base_smiles:
            raise ValueError("SMILES is empty.")
        ligands_raw.append({"name": output_name or "ligand", "base_smiles": base_smiles})

    elif input_type == "SMI_FILE":
        if not uploaded_bytes:
            raise ValueError("No .smi uploaded.")
        text = uploaded_bytes.decode("utf-8", errors="replace")
        for smi, name in parse_smi_lines(text):
            ligands_raw.append({"name": name, "base_smiles": smi})

    elif input_type == "FILE":
        if not uploaded_bytes or not uploaded_name:
            raise ValueError("No ligand file uploaded.")
        ext = os.path.splitext(uploaded_name)[1].lower()
        tmp_path = out / f"uploaded{ext}"
        tmp_path.write_bytes(uploaded_bytes)
        mol_in = None
        if ext == ".pdb":
            mol_in = Chem.MolFromPDBFile(str(tmp_path), removeHs=False, sanitize=False)
        elif ext == ".mol2":
            mol_in = Chem.MolFromMol2File(str(tmp_path), removeHs=False, sanitize=False)
        elif ext == ".sdf":
            supplier = Chem.SDMolSupplier(str(tmp_path), removeHs=False, sanitize=False)
            mol_in = next((m for m in supplier if m is not None), None)
        else:
            raise ValueError("Unsupported file type.")
        if mol_in is None:
            raise ValueError("RDKit could not parse uploaded ligand.")
        try:
            frags = Chem.GetMolFrags(mol_in, asMols=True, sanitizeFrags=False)
            if len(frags) > 1:
                mol_in = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            Chem.SanitizeMol(mol_in)
        except Exception:
            pass
        base_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol_in), canonical=True)
        ligands_raw.append({"name": output_name or os.path.splitext(uploaded_name)[0],
                             "base_smiles": base_smiles})
    else:
        raise ValueError("Unknown input_type")

    # ── Enumerate stereoisomers ──────────────────────────────────────────────
    keep_stereo = not enumerate_stereoisomers
    ligands: List[dict] = []
    for lig in ligands_raw:
        ligands.extend(generate_RS_variants(lig["base_smiles"], lig["name"], keep_stereo))

    results: List[dict] = []
    format_warnings: List[str] = []

    for lig in ligands:
        base_name  = lig["name"]
        stereo     = lig.get("stereo")
        suffix     = f"_{stereo}" if stereo else ""
        pretty_name = base_name + suffix
        base_smiles = lig["base_smiles"]

        # Standardize
        can_smi, status = standardize_smiles(base_smiles)
        if can_smi is None:
            print(f"Standardization failed for {pretty_name}: {status}")
            can_smi = base_smiles  # fallback
        else:
            base_smiles = can_smi

        # PubChem lookup (optional)
        pc_result: dict = {}
        if use_pubchem and _REQUESTS_OK:
            try:
                pc_result = pubchem_lookup(base_smiles)
            except Exception as e:
                print(f"PubChem lookup failed: {e}")

        # Generate ranked microstates
        try:
            (top_microstates, ambiguous, all_microstates,
             tr_flag, tr_motifs, ml_preds, pka_scalar, pka_source) = generate_ranked_microstates(
                base_smiles,
                target_ph      = target_pH,
                ph_window      = ph_window,
                max_tautomers  = max_tautomers,
                top_n          = top_n_microstates,
                pubchem_result = pc_result,
                use_iupac_pka  = use_iupac_pka,
            )
        except Exception as e:
            print(f"Microstate generation failed for {pretty_name}: {e}")
            format_warnings.append(f"Microstate generation failed for {pretty_name}: {e}")
            continue

        if not top_microstates:
            format_warnings.append(f"No valid microstates for {pretty_name}")
            continue

        top_state  = top_microstates[0]
        ph_smiles  = top_state["microstate_smiles"]
        formal_charge = top_state["net_charge"]
        cp         = get_charge_profile(ph_smiles)

        # Build 3D structures for top-k microstates
        saved_3d: List[dict] = []
        for row in top_microstates[:max(1, write_alt_3d_for_top_k)]:
            rk  = row["microstate_rank"]
            bp  = str(out / f"{base_name}{suffix}_micro{rk}_min")
            try:
                m3d = build_minimized_3d(row["microstate_smiles"])
                save_result = save_molecule_files(m3d, bp, formats_to_save)
                files = save_result["files"]
                # 2D image
                png_path = str(out / f"{base_name}{suffix}_micro{rk}_2D.png")
                save_2d_structure_image(row["microstate_smiles"], png_path)
                if Path(png_path).exists():
                    files["png_2d"] = png_path
                for w in save_result["warnings"]:
                    if w not in format_warnings:
                        format_warnings.append(w)
                saved_3d.append({"rank": rk, "files": files,
                                  "smiles": row["microstate_smiles"]})
            except Exception as e:
                print(f"3D build failed rank {rk}: {e}")

        # Microstate CSV (all microstates, not just top-k)
        micro_csv_path = str(out / f"{base_name}{suffix}_microstates.csv")
        try:
            import pandas as pd
            micro_df = pd.DataFrame(
                [{k: v for k, v in r.items() if k != "charged_atom_rows"}
                 for r in all_microstates]
            )
            micro_df.to_csv(micro_csv_path, index=False)
        except Exception as e:
            print(f"Microstate CSV write failed: {e}")
            micro_csv_path = None

        # Build result entry
        rank1_files = saved_3d[0]["files"] if saved_3d else {}
        result_entry = {
            "name":             pretty_name,
            "base_smiles":      base_smiles,
            "ph_smiles":        ph_smiles,
            "pka_pred":         pka_scalar,
            "pka_source":       pka_source,
            "formal_charge":    formal_charge,
            "has_pos":          cp["has_pos"],
            "has_neg":          cp["has_neg"],
            "n_pos_atoms":      cp["n_pos_atoms"],
            "n_neg_atoms":      cp["n_neg_atoms"],
            "is_zwitterion":    cp["is_zwitterion_strict"],
            "charged_atoms":    charged_atoms_text(cp),
            # New v4 fields
            "ambiguous":        ambiguous,
            "tautomer_rich":    tr_flag,
            "tautomer_motifs":  tr_motifs,
            "selection_score":  top_state["selection_score"],
            "decision_backend": top_state["decision_backend"],
            "flag_amide_preserved":              top_state.get("flag_amide_preserved", False),
            "flag_imidic_acid_penalty":          top_state.get("flag_imidic_acid_penalty", False),
            "flag_amide_n_deprotonation":        top_state.get("flag_amide_n_deprotonation_penalty", False),
            "flag_borderline_pka":               top_state.get("flag_borderline_pka", False),
            "pubchem_cid":      pc_result.get("cid"),
            "pubchem_pka":      pc_result.get("pka_values", []),
            "pubchem_confidence": pc_result.get("confidence", "n/a"),
            "top_microstates":  [{k: v for k, v in r.items() if k != "charged_atom_rows"}
                                  for r in top_microstates],
            "n_all_microstates": len(all_microstates),
            "microstate_csv":   micro_csv_path,
            # 3D file paths (rank-1)
            "minimized_pdb":    rank1_files.get("pdb"),
            "minimized_sdf":    rank1_files.get("sdf"),
            "minimized_mol2":   rank1_files.get("mol2"),
            # alt 3D
            "alt_3d":           saved_3d,
        }
        if stereo:
            result_entry["stereoisomer_id"] = stereo

        results.append(result_entry)

    # ── Summary ──────────────────────────────────────────────────────────────
    summary_lines = [
        "=" * 80,
        "pKaNET Cloud v4 — Analysis Summary",
        "=" * 80,
        f"Target pH            : {target_pH}",
        f"pH window            : ±{ph_window/2:.1f}",
        f"Max tautomers        : {max_tautomers}",
        f"Top microstates      : {top_n_microstates}",
        f"Stereo enumeration   : {'Enabled' if enumerate_stereoisomers else 'Disabled'}",
        f"Total structures     : {len(results)}",
        f"pKa backend          : IUPAC → PubChem → pKaPredict (ML) → heuristic",
        "=" * 80,
        "",
    ]
    for r in results:
        summary_lines.append(f"Molecule: {r['name']}")
        summary_lines.append("-" * 80)
        summary_lines.append(f"  Base SMILES              : {r['base_smiles']}")
        summary_lines.append(f"  Selected SMILES (rank-1) : {r['ph_smiles']}")
        pka_str = f"{r['pka_pred']:.2f} ({r['pka_source']})" if r["pka_pred"] is not None else "N/A"
        summary_lines.append(f"  Predicted pKa            : {pka_str}")
        if r["pubchem_pka"]:
            summary_lines.append(f"  PubChem pKa              : {r['pubchem_pka']} (confidence={r['pubchem_confidence']})")
        summary_lines.append(f"  Formal Charge (pH {target_pH})  : {r['formal_charge']:+d}")
        summary_lines.append(f"  Charged atoms            : {r['charged_atoms']}")
        summary_lines.append(f"  Zwitterion (strict)      : {'YES' if r.get('is_zwitterion') else 'NO'}")
        summary_lines.append(f"  Ambiguous assignment     : {'YES' if r.get('ambiguous') else 'NO'}")
        summary_lines.append(f"  Tautomer-rich            : {'YES' if r.get('tautomer_rich') else 'NO'}")
        summary_lines.append(f"  Selection score (rank-1) : {r.get('selection_score', 'N/A')}")
        summary_lines.append(f"  Decision backend         : {r.get('decision_backend', 'N/A')}")
        summary_lines.append(f"  Microstates evaluated    : {r.get('n_all_microstates', 'N/A')}")
        summary_lines.append("")
    summary_lines += ["=" * 80, "pKaNET Cloud v4", "=" * 80]
    summary_text = "\n".join(summary_lines).strip()
    (out / "summary.txt").write_text(summary_text + "\n")

    # Processing log for batch runs
    if input_type == "SMI_FILE" and results:
        log_lines = [
            "# pKaNET Cloud v4 — Processing Log",
            f"# Target pH: {target_pH}",
            f"# Stereo enumeration: {'enabled' if enumerate_stereoisomers else 'disabled'}",
            f"# Total molecules: {len(results)}",
            "#" + "=" * 70,
            "",
            "# Name | pH-SMILES | Charge | pKa | pKa Source | Zwitterion | Ambiguous | PubChem_pKa",
            "",
        ]
        for r in results:
            pka_str = f"{r['pka_pred']:.2f}" if r["pka_pred"] is not None else "N/A"
            zw  = "Yes" if r.get("is_zwitterion") else "No"
            amb = "Yes" if r.get("ambiguous") else "No"
            pc  = str(r.get("pubchem_pka", []))
            log_lines.append(
                f"{r['name']}\t{r['ph_smiles']}\t{r['formal_charge']:+d}\t"
                f"{pka_str}\t{r['pka_source']}\t{zw}\t{amb}\t{pc}"
            )
        (out / "processing.log").write_text("\n".join(log_lines) + "\n")

    return {
        "results":         results,
        "summary_text":    summary_text,
        "out_dir":         str(out),
        "format_warnings": format_warnings,
    }

# ─────────────────────────────────────────────────────────────────────────────
# ZIP helpers
# ─────────────────────────────────────────────────────────────────────────────

def zip_minimized_structures(out_dir: str, zip_path: str, selected_formats: List[str]) -> str:
    out = Path(out_dir)
    zp  = Path(zip_path)
    formats_lower = [fmt.lower() for fmt in selected_formats]
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.glob("*_min.*"):
            suffix = p.suffix.lower()
            if suffix == ".pdb" and "pdb" in formats_lower:
                z.write(p, arcname=p.name)
            elif suffix == ".mol2" and "mol2" in formats_lower:
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
