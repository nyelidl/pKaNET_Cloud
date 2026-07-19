#!/usr/bin/env python3
"""
pkanet-cloud.py — pKaNET Cloud+ wrapper for Anyone Can Dock Colab notebook.

This module re-exports the three functions that the ACD Ligand Preparation
cell expects:

    standardize_smiles(smiles)
        → (canonical_smiles | None, message_str)

    pubchem_lookup(smiles)
        → dict with keys: available, cid, inchikey, pka_values,
                          source_texts, flags, confidence, error

    generate_ranked_microstates(
        base_smiles, target_ph=7.4, ph_window=1.0,
        max_tautomers=8, top_n=5, pubchem_result=None,
    )
        → (top_list, ambiguous_bool, all_microstates, tautomer_rich_flag,
           tautomer_motifs, ml_predictions)

Each top_list entry is a dict with at least:
    microstate_smiles, selection_score, pKa_source, ...

Usage (in Colab):
    1.  Upload both `pkanet-cloud.py` and `core.py` to the working directory.
    2.  Set `pkanet_cloud_path = "./pkanet-cloud.py"` in the Ligand Prep cell.
    3.  The cell will `importlib`-load this file and call the three functions.

Engine:  pKaNET v81 — Hammett/Taft substituent pKa corrections
         (core.py must be co-located in the same directory)
"""

__version__ = "81.0.0"

# ── Resolve core.py relative to this file's location ─────────────────────────
import importlib.util as _iu
import sys as _sys
from pathlib import Path as _P

_HERE = _P(__file__).resolve().parent
_CORE = _HERE / "core.py"

if not _CORE.exists():
    # Fallback: try current working directory
    _CORE = _P("core.py")
    if not _CORE.exists():
        raise ImportError(
            f"❌ pkanet-cloud.py requires core.py in the same directory.\n"
            f"   Searched: {_HERE / 'core.py'} and ./core.py"
        )

_spec = _iu.spec_from_file_location("pkanet_core", str(_CORE))
_core = _iu.module_from_spec(_spec)
_sys.modules["pkanet_core"] = _core
_spec.loader.exec_module(_core)


# ── Re-export the three API functions ────────────────────────────────────────
standardize_smiles          = _core.standardize_smiles
pubchem_lookup              = _core.pubchem_lookup
generate_ranked_microstates = _core.generate_ranked_microstates


# ── Optional: additional helpers the notebook might use ──────────────────────
# These are not required by the Ligand Prep cell but are useful for
# standalone scripting or debugging.

heuristic_net_charge  = _core.heuristic_net_charge
find_ionizable_sites  = _core.find_ionizable_sites
get_charge_profile    = _core.get_charge_profile
enumerate_and_filter_tautomers = _core.enumerate_and_filter_tautomers

# Quick smoke test when run directly
if __name__ == "__main__":
    print(f"pKaNET Cloud+ v{__version__}")
    print(f"Engine: {_CORE}")
    print()

    test_cases = [
        ("aspirin",        "CC(=O)Oc1ccccc1C(=O)O"),
        ("glycine",        "NCC(=O)O"),
        ("erlotinib",      "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"),
        ("phenylhydrazine","NNc1ccccc1"),
        ("2,4-dinitrophenol", "O=[N+]([O-])c1ccc(O)c([N+](=O)[O-])c1"),
    ]

    for name, smi in test_cases:
        charge = heuristic_net_charge(smi, 7.4)
        print(f"  {name:25s}  charge@7.4 = {charge:+d}   {smi}")

    print()
    print("Microstate generation test (glycine):")
    top, amb, all_ms, tr, motifs, ml = generate_ranked_microstates(
        "NCC(=O)O", target_ph=7.4, top_n=3,
    )
    for i, ms in enumerate(top[:3]):
        print(f"  #{i+1}  {ms['microstate_smiles']:30s}  "
              f"score={ms.get('selection_score', '?')}")
    print(f"  ambiguous={amb}, tautomer_rich={tr}")
