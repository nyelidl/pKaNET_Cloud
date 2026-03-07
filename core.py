# core.py
from __future__ import annotations
from pathlib import Path
import os
import zipfile
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from dimorphite_dl import protonate_smiles
from pkapredict import load_model, predict_pKa
import subprocess
import shutil


# Check if Open Babel is available
_OBABEL_AVAILABLE = None

def check_obabel():
    """Check if obabel command is available"""
    global _OBABEL_AVAILABLE
    if _OBABEL_AVAILABLE is None:
        _OBABEL_AVAILABLE = shutil.which("obabel") is not None
    return _OBABEL_AVAILABLE


def convert_pdb_to_mol2_obabel(pdb_path: str, mol2_path: str) -> bool:
    """
    Convert PDB to MOL2 using Open Babel
    
    Args:
        pdb_path: Path to input PDB file
        mol2_path: Path to output MOL2 file
    
    Returns:
        True if conversion successful, False otherwise
    """
    if not check_obabel():
        return False
    
    try:
        # Run obabel conversion
        result = subprocess.run(
            ["obabel", pdb_path, "-O", mol2_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Check if conversion was successful
        if result.returncode == 0 and Path(mol2_path).exists():
            return True
        else:
            print(f"Open Babel conversion failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("Open Babel conversion timed out")
        return False
    except Exception as e:
        print(f"Open Babel conversion error: {e}")
        return False


# =========================================
# Charge profile helper (net charge + zwitterion flag)
# =========================================
def charge_profile_from_smiles(smiles: str) -> Dict[str, Any]:
    """
    Return:
      - net_charge
      - has_pos / has_neg (any + / any − atoms)
      - strict zwitterion: has_pos and has_neg and net_charge == 0
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES for charge profiling.")
    atom_charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    n_pos = sum(c > 0 for c in atom_charges)
    n_neg = sum(c < 0 for c in atom_charges)
    net = int(sum(atom_charges))
    return {
        "net_charge": net,
        "has_pos": n_pos > 0,
        "has_neg": n_neg > 0,
        "n_pos_atoms": int(n_pos),
        "n_neg_atoms": int(n_neg),
        "is_zwitterion_strict": bool((n_pos > 0) and (n_neg > 0) and (net == 0)),
    }


# =========================================
# IUPAC pKa lookup (optional - requires pandas and network access)
# =========================================
_IUPAC_DF = None
_PKA_MAP_ALL = None
_IUPAC_LOADED = False

def load_iupac_dataset():
    """Load IUPAC pKa dataset if pandas is available"""
    global _IUPAC_DF, _PKA_MAP_ALL, _IUPAC_LOADED
    
    if _IUPAC_LOADED:
        return
    
    _IUPAC_LOADED = True
    
    try:
        import pandas as pd
        IUPAC_CSV_URL = "https://raw.githubusercontent.com/IUPAC/Dissociation-Constants/main/iupac_high-confidence_v2_3.csv"
        
        print("⏳ Loading IUPAC pKa dataset...")
        iupac_df = pd.read_csv(IUPAC_CSV_URL)
        print(f"✓ IUPAC dataset loaded: {len(iupac_df):,} rows")
        
        # Find SMILES and pKa columns
        cols = list(iupac_df.columns)
        lower_map = {c.lower(): c for c in cols}
        
        smiles_col = None
        for w in ["SMILES", "smiles"]:
            if w in cols:
                smiles_col = w
                break
            if w.lower() in lower_map:
                smiles_col = lower_map[w.lower()]
                break
        
        pka_col = None
        for w in ["pka_value", "pKa", "pka", "value"]:
            if w in cols:
                pka_col = w
                break
            if w.lower() in lower_map:
                pka_col = lower_map[w.lower()]
                break
        
        if smiles_col is None or pka_col is None:
            print(f"⚠️ Cannot find SMILES/pKa columns in IUPAC dataset")
            return
        
        # Build canonical SMILES lookup
        def canonicalize(smi):
            m = Chem.MolFromSmiles(str(smi).strip())
            return Chem.MolToSmiles(m, canonical=True) if m else None
        
        iupac_df["_cansmi"] = iupac_df[smiles_col].apply(canonicalize)
        
        pka_map = defaultdict(list)
        for csmi, pka in zip(iupac_df["_cansmi"], iupac_df[pka_col]):
            if csmi is None:
                continue
            try:
                pka_map[csmi].append(float(pka))
            except Exception:
                pass
        
        _IUPAC_DF = iupac_df
        _PKA_MAP_ALL = pka_map
        print(f"✓ IUPAC indexed molecules: {len(pka_map):,}")
        
    except Exception as e:
        print(f"⚠️ IUPAC dataset load failed: {e}")
        print("   Will use pKaPredict only.")


def lookup_pka_iupac_stats(query_smiles: str) -> Optional[Dict[str, Any]]:
    """Return dict with median/mean/min/max/n/all if matched; else None."""
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
        "pka_mean": float(np.mean(vals)),
        "pka_min": float(np.min(vals)),
        "pka_max": float(np.max(vals)),
        "n": len(vals),
        "all": vals,
        "canonical_smiles": canonical_smi,
    }


# Load model once (cached in module)
_PKANET_MODEL = None
_DESCRIPTOR_NAMES = None

def get_model():
    global _PKANET_MODEL, _DESCRIPTOR_NAMES
    if _PKANET_MODEL is None:
        _PKANET_MODEL = load_model()
        
        # Get descriptor names from model
        if hasattr(_PKANET_MODEL, 'feature_name_'):
            _DESCRIPTOR_NAMES = _PKANET_MODEL.feature_name_
            print(f"✓ Model loaded with {len(_DESCRIPTOR_NAMES)} descriptors")
        else:
            # Fallback: use first N descriptors from RDKit
            from rdkit.Chem import Descriptors
            all_descriptors = [desc[0] for desc in Descriptors._descList]
            _DESCRIPTOR_NAMES = all_descriptors[:_PKANET_MODEL.n_features_]
            print(f"✓ Model loaded, using {len(_DESCRIPTOR_NAMES)} RDKit descriptors")
    
    return _PKANET_MODEL, _DESCRIPTOR_NAMES


def predict_pka_pkanet(smiles: str) -> float:
    """
    Predict pKa using pKaPredict ML model
    
    Args:
        smiles: SMILES string
    
    Returns:
        Predicted pKa value as float
    """
    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES for pKa prediction.")
    
    try:
        # Get model and descriptor names
        model, descriptor_names = get_model()
        
        # Predict pKa using correct API: predict_pKa(smiles, model, descriptor_names)
        pka_value = predict_pKa(smiles, model, descriptor_names)
        
        print(f"✓ pKa prediction successful: {pka_value:.2f}")
        
        # Handle both single value and array returns
        if isinstance(pka_value, (list, tuple)):
            pka_value = pka_value[0]
        elif hasattr(pka_value, '__iter__') and not isinstance(pka_value, str):
            pka_value = next(iter(pka_value))
            
        return float(pka_value)
        
    except Exception as e:
        print(f"Error during pKa prediction for SMILES '{smiles}': {e}")
        raise


def get_pka_iupac_else_ml(smiles: str) -> Tuple[float, str, Optional[Dict]]:
    """
    Get pKa value: try IUPAC first, then fall back to pKaPredict ML
    
    Returns:
        (pka_value, source_string, iupac_stats_dict_or_None)
    """
    stats = lookup_pka_iupac_stats(smiles)
    if stats is not None:
        return stats["pka_median"], f"IUPAC (n={stats['n']})", stats
    return predict_pka_pkanet(smiles), "pKaPredict (ML)", None


def ph_adjust_smiles_dimorphite(smiles_str: str, ph: float, mode: str = "AUTO") -> Tuple[str, int, Dict[str, Any]]:
    """
    Generate protonated SMILES at target pH using Dimorphite-DL.
    
    Args:
        smiles_str: Input SMILES
        ph: Target pH
        mode: Charge selection mode
            - AUTO: return first variant (Dimorphite default)
            - FORCE_ZWITTERION: return a strict zwitterion if present among variants; else fallback to most neutral
            - NORMAL: choose most neutral (smallest |net charge|)
    
    Returns:
        (selected_smiles, net_charge, charge_profile_dict)
    """
    prot_list = protonate_smiles(smiles_str, ph_min=ph, ph_max=ph, max_variants=4)
    if not prot_list:
        raise ValueError("Dimorphite-DL returned no protonation state.")

    candidates = []
    for smi in prot_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        prof = charge_profile_from_smiles(smi)
        candidates.append((smi, prof))

    if not candidates:
        raise ValueError("RDKit could not parse Dimorphite-DL variants.")

    if mode == "FORCE_ZWITTERION":
        # First try to find a strict zwitterion
        for smi, prof in candidates:
            if prof["is_zwitterion_strict"]:
                return smi, prof["net_charge"], prof
        # fallback: most neutral
        smi, prof = min(candidates, key=lambda x: abs(x[1]["net_charge"]))
        return smi, prof["net_charge"], prof

    if mode == "NORMAL":
        # Choose most neutral (smallest |net charge|)
        smi, prof = min(candidates, key=lambda x: abs(x[1]["net_charge"]))
        return smi, prof["net_charge"], prof

    # AUTO - return first variant
    smi, prof = candidates[0]
    return smi, prof["net_charge"], prof


def build_minimized_3d(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES for 3D build.")
    mol = Chem.AddHs(mol)

    code = -1
    try:
        try:
            params = AllChem.ETKDGv3()
        except AttributeError:
            params = AllChem.ETKDG()
        params.randomSeed = 0xF00D
        code = AllChem.EmbedMolecule(mol, params)
    except Exception:
        code = AllChem.EmbedMolecule(mol, randomSeed=0xF00D, maxAttempts=2000)

    if code != 0 or mol.GetNumConformers() == 0:
        code2 = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=0xF00D, maxAttempts=2000)
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

def parse_smi_lines(text: str):
    records = []
    idx = 1
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        smi = parts[0]
        name = parts[1] if len(parts) > 1 else f"mol_{idx:03d}"
        records.append((smi, name))
        idx += 1
    return records

def generate_RS_variants(base_smiles: str, base_name: str, keep_original: bool = False):
    """
    Generate stereoisomer variants.
    
    Args:
        base_smiles: Input SMILES
        base_name: Base molecule name
        keep_original: If True, return only the original stereochemistry
    
    Returns:
        List of dicts with name, stereo, base_smiles
    """
    mol = Chem.MolFromSmiles(base_smiles)
    if mol is None:
        return [{"name": base_name, "stereo": None, "base_smiles": base_smiles}]

    if keep_original:
        return [{"name": base_name, "stereo": None, "base_smiles": Chem.MolToSmiles(mol, isomericSmiles=True)}]

    opts = StereoEnumerationOptions(onlyUnassigned=False)
    isomers = list(EnumerateStereoisomers(mol, options=opts))

    if len(isomers) == 1:
        return [{"name": base_name, "stereo": None, "base_smiles": Chem.MolToSmiles(isomers[0], isomericSmiles=True)}]

    variants = []
    used = set()
    for iso in isomers:
        Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
        centers = Chem.FindMolChiralCenters(iso, includeUnassigned=False)
        labs = {lab for _, lab in centers if lab in ("R", "S")}
        
        # Label for filename
        label_here = None
        if "R" in labs and "R" not in used:
            label_here = "R"
        elif "S" in labs and "S" not in used:
            label_here = "S"

        if label_here:
            used.add(label_here)
            variants.append({
                "name": base_name,
                "stereo": label_here,
                "base_smiles": Chem.MolToSmiles(iso, isomericSmiles=True)
            })

        if used == {"R", "S"}:
            break

    return variants if variants else [{"name": base_name, "stereo": None, "base_smiles": Chem.MolToSmiles(isomers[0], isomericSmiles=True)}]


def save_2d_structure_image(smiles: str, output_path: str, size=(800, 600)) -> bool:
    """
    Save 2D structure as PNG image
    
    Args:
        smiles: SMILES string
        output_path: Path to save PNG file
        size: Image size (width, height)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from rdkit.Chem import Draw
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=size)
        img.save(output_path)
        return True
        
    except (ImportError, OSError, AttributeError) as e:
        print(f"Warning: Could not generate 2D structure image: {e}")
        return False
    except Exception as e:
        print(f"Warning: 2D structure image generation failed: {e}")
        return False


def save_molecule_files(mol, base_path: str, formats: List[str]) -> Dict[str, Any]:
    """
    Save molecule to multiple file formats.
    Always generates SDF for visualization. User-selected formats are also saved.
    If MOL2 is requested but RDKit doesn't support it, tries to convert from PDB using Open Babel.
    
    Args:
        mol: RDKit molecule object
        base_path: Base file path without extension
        formats: List of formats to save (e.g., ["PDB", "MOL2"])
    
    Returns:
        Dictionary with 'files' (mapping format to file path) and 'warnings' (list of warnings)
    """
    saved_files = {}
    warnings = []
    mol2_requested = "MOL2" in [f.upper() for f in formats]
    mol2_via_obabel = False
    
    # Always save SDF first (for visualization)
    try:
        sdf_path = f"{base_path}.sdf"
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        saved_files["sdf"] = sdf_path
    except Exception as e:
        warnings.append(f"Could not save SDF format: {e}")
        print(f"Warning: Could not save SDF format: {e}")
    
    # Now save user-requested formats
    for fmt in formats:
        fmt_upper = fmt.upper()
        
        # Skip SDF if already saved
        if fmt_upper == "SDF":
            continue
            
        try:
            if fmt_upper == "PDB":
                file_path = f"{base_path}.pdb"
                Chem.MolToPDBFile(mol, file_path)
                saved_files["pdb"] = file_path
                
            elif fmt_upper == "MOL2":
                file_path = f"{base_path}.mol2"
                
                # Try RDKit first
                if hasattr(Chem, 'MolToMol2File'):
                    try:
                        Chem.MolToMol2File(mol, file_path)
                        saved_files["mol2"] = file_path
                        continue
                    except Exception as e:
                        print(f"RDKit MOL2 failed, will try Open Babel: {e}")
                
                # RDKit MOL2 not available, try Open Babel conversion
                if "pdb" not in saved_files:
                    # Need to generate PDB first for conversion
                    pdb_path = f"{base_path}.pdb"
                    try:
                        Chem.MolToPDBFile(mol, pdb_path)
                        saved_files["pdb"] = pdb_path
                    except Exception as e:
                        warnings.append(f"Could not generate PDB for MOL2 conversion: {e}")
                        continue
                
                # Try converting PDB to MOL2 with Open Babel
                pdb_path = saved_files.get("pdb")
                if pdb_path and convert_pdb_to_mol2_obabel(pdb_path, file_path):
                    saved_files["mol2"] = file_path
                    mol2_via_obabel = True
                else:
                    if not check_obabel():
                        warnings.append("MOL2 format not available. Install Open Babel (obabel) to enable MOL2 output.")
                    else:
                        warnings.append("MOL2 conversion failed. Using PDB format instead.")
        
        except Exception as e:
            warnings.append(f"Could not save {fmt_upper} format: {e}")
            print(f"Warning: Could not save {fmt_upper} format: {e}")
            continue
    
    # Add info message if MOL2 was generated via Open Babel
    if mol2_via_obabel:
        warnings.append("ℹ️ MOL2 files generated using Open Babel (converted from PDB)")
    
    return {"files": saved_files, "warnings": warnings}


def run_job(
    *,
    input_type: str,
    smiles_text: str | None,
    uploaded_bytes: bytes | None,
    uploaded_name: str | None,
    target_pH: float,
    output_name: str,
    out_dir: str,
    output_formats: List[str] = None,
    enumerate_stereoisomers: bool = True,
    charge_mode: str = "AUTO",
    use_iupac_pka: bool = True,
    use_xtb_pka: bool = False,
    xtb_results_map: dict | None = None,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Default to PDB if no formats specified
    if output_formats is None or len(output_formats) == 0:
        output_formats = ["PDB"]
    
    # User-selected formats (SDF is handled separately, always generated)
    formats_to_save = [fmt.upper() for fmt in output_formats]
    
    # Load IUPAC dataset if requested
    if use_iupac_pka and not _IUPAC_LOADED:
        try:
            load_iupac_dataset()
        except Exception:
            pass

    ligands_raw = []

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
            raise ValueError("Unsupported file type. Use .pdb, .mol2, or .sdf")

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
        ligands_raw.append({"name": output_name or os.path.splitext(uploaded_name)[0], "base_smiles": base_smiles})

    else:
        raise ValueError("Unknown input_type")

    # Enumerate stereoisomers if requested
    ligands = []
    keep_stereo = not enumerate_stereoisomers
    for lig in ligands_raw:
        ligands.extend(generate_RS_variants(lig["base_smiles"], lig["name"], keep_stereo))

    results = []
    format_warnings = []  # Collect warnings across all molecules
    
    for lig in ligands:
        base_name = lig["name"]
        stereo = lig.get("stereo")
        suffix = f"_{stereo}" if stereo else ""
        pretty_name = base_name + suffix

        base_smiles = lig["base_smiles"]

        # ── pKa Prediction ──────────────────────────────────────────────
        # Logic:
        #   IUPAC checked, xTB off  → IUPAC if matched, else ML
        #   IUPAC checked, xTB on   → IUPAC if matched (xTB shown alongside in app)
        #                              if no IUPAC match → None (xTB only, skip ML)
        #   IUPAC off,     xTB on   → None (xTB only, skip ML)
        #   IUPAC off,     xTB off  → ML only
        pka_pred          = None
        pka_source        = None
        pka_n             = None
        pka_iupac_matched = False

        try:
            if use_iupac_pka:
                stats = lookup_pka_iupac_stats(base_smiles)
                if stats is not None:
                    pka_pred          = stats["pka_median"]
                    pka_source        = f"IUPAC (n={stats['n']})"
                    pka_n             = stats["n"]
                    pka_iupac_matched = True
                    print(f"pKa (IUPAC, n={stats['n']}) for {pretty_name}: {pka_pred:.2f}")

            if pka_pred is None:
                # No IUPAC match (or IUPAC disabled)
                if use_xtb_pka:
                    # Skip ML — xTB result will be the sole pKa source
                    pka_source = "xTB"
                    print(f"No IUPAC match for {pretty_name} — will use xTB pKa only.")
                else:
                    pka_pred   = predict_pka_pkanet(base_smiles)
                    pka_source = "pKaPredict (ML)"
                    print(f"pKa (pKaPredict ML) for {pretty_name}: {pka_pred:.2f}")

        except Exception as e:
            print(f"Warning: pKa prediction failed for {pretty_name}: {e}")
            warning_msg = f"pKa prediction failed for {pretty_name}: {str(e)}"
            if warning_msg not in format_warnings:
                format_warnings.append(warning_msg)

        # pH adjustment with charge mode
        try:
            ph_smiles, formal_charge, charge_prof = ph_adjust_smiles_dimorphite(base_smiles, target_pH, charge_mode)
        except Exception as e:
            print(f"Error: pH adjustment failed for {pretty_name}: {e}")
            continue

        mol_min = build_minimized_3d(ph_smiles)

        # Save molecule in requested formats (SDF always included)
        base_file_path = str(out / f"{base_name}{suffix}_min")
        save_result = save_molecule_files(mol_min, base_file_path, formats_to_save)
        saved_files = save_result["files"]
        
        # Save 2D structure as PNG for visualization/download
        png_path = str(out / f"{base_name}{suffix}_2D.png")
        if save_2d_structure_image(ph_smiles, png_path):
            saved_files["png_2d"] = png_path
        
        # Collect unique warnings
        for warning in save_result["warnings"]:
            if warning not in format_warnings:
                format_warnings.append(warning)

        result_entry = {
            "name": pretty_name,
            "base_smiles": base_smiles,
            "ph_smiles": ph_smiles,
            "pka_pred": pka_pred,
            "pka_source": pka_source,
            "pka_n": pka_n,
            "pka_iupac_matched":  pka_iupac_matched,
            "formal_charge": formal_charge,
            "has_pos": charge_prof["has_pos"],
            "has_neg": charge_prof["has_neg"],
            "n_pos_atoms": charge_prof["n_pos_atoms"],
            "n_neg_atoms": charge_prof["n_neg_atoms"],
            "is_zwitterion": charge_prof["is_zwitterion_strict"],
        }
        
        # Add stereoisomer ID if it was enumerated
        if stereo:
            result_entry["stereoisomer_id"] = stereo
        
        # Add file paths to result
        if "pdb" in saved_files:
            result_entry["minimized_pdb"] = saved_files["pdb"]
        if "sdf" in saved_files:
            result_entry["minimized_sdf"] = saved_files["sdf"]
        if "mol2" in saved_files:
            result_entry["minimized_mol2"] = saved_files["mol2"]
        
        results.append(result_entry)

    # Write summary file
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("pKaNET Cloud - Analysis Summary")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Target pH: {target_pH}")
    summary_lines.append(f"Charge mode: {charge_mode}")
    summary_lines.append(f"Stereoisomer enumeration: {'Enabled' if enumerate_stereoisomers else 'Disabled'}")
    summary_lines.append(f"Total structures generated: {len(results)}")
    summary_lines.append(f"pKa: {'IUPAC (if matched) → else pKaPredict (ML)' if use_iupac_pka else 'pKaPredict (ML)'}")
    summary_lines.append("Zwitterion (strict): has + and − atoms AND net charge = 0")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    for r in results:
        summary_lines.append(f"Molecule: {r['name']}")
        summary_lines.append("-" * 80)
        summary_lines.append(f"  Base SMILES          : {r['base_smiles']}")
        summary_lines.append(f"  pH-adjusted SMILES   : {r['ph_smiles']}")
        
# Format pKa value safely
        if r['pka_pred'] is not None:
            pka_str = f"{r['pka_pred']:.2f} ({r['pka_source']})"
            summary_lines.append(f"  Predicted pKa        : {pka_str}")
        elif r.get('pka_source') == "xTB":
            summary_lines.append(f"  Predicted pKa        : see xTB results below")
        else:
            summary_lines.append(f"  Predicted pKa        : N/A")

        # xTB pKa results (if xtb_results_map was passed in)
        if xtb_results_map:
            base_smi = r["base_smiles"]
            xtb_res  = xtb_results_map.get(base_smi, [])
            for xr in xtb_res:
                if xr["pKa"] is not None:
                    summary_lines.append(
                        f"  xTB pKa ({xr['group']:8s})  : {xr['pKa']:.1f}  "
                        f"[ΔE={xr['dE_kcal']:+.3f} kcal/mol]"
                    )
```

This block sits immediately **after** the zwitterion/charge lines and **before** the `generated_formats` block, so the final order in the summary for each molecule is:
```
  Predicted pKa        : 4.75 (IUPAC n=3)     ← or "see xTB results below"
  xTB pKa (acid    )   : 4.6  [ΔE=−0.203 kcal/mol]
  Formal Charge        : 0
  Zwitterion           : NO
  Output Formats       : PDB, SDF
            
        summary_lines.append(f"  Formal Charge (pH {target_pH}): {r['formal_charge']:+d}")
        summary_lines.append(f"  Zwitterion (strict)  : {'YES' if r.get('is_zwitterion') else 'NO'}")
        summary_lines.append(f"  + atoms / − atoms    : {r.get('n_pos_atoms', 0)} / {r.get('n_neg_atoms', 0)}")
        if xtb_results_map:
            base_smi = r["base_smiles"]
            xtb_res  = xtb_results_map.get(base_smi, [])
            for xr in xtb_res:
                if xr["pKa"] is not None:
                    summary_lines.append(
                        f"  xTB pKa ({xr['group']:8s})  : {xr['pKa']:.1f}  "
                        f"[ΔE={xr['dE_kcal']:+.3f} kcal/mol]"
                    )
        # Show what formats were actually generated
        generated_formats = []
        if "minimized_pdb" in r:
            generated_formats.append("PDB")
        if "minimized_mol2" in r:
            generated_formats.append("MOL2")
        if "minimized_sdf" in r:
            generated_formats.append("SDF")
        summary_lines.append(f"  Output Formats       : {', '.join(generated_formats)}")
        
        if "stereoisomer_id" in r:
            summary_lines.append(f"  Stereoisomer         : {r['stereoisomer_id']}")
        summary_lines.append("")
    
    summary_lines.append("=" * 80)
    summary_lines.append("pKa: IUPAC (if matched) → pKaPredict (ML-based)")
    summary_lines.append("=" * 80)
    
    summary_text = "\n".join(summary_lines).strip()
    (out / "summary.txt").write_text(summary_text + "\n")
    
    # Create log file for SMI_FILE input
    if input_type == "SMI_FILE" and len(results) > 0:
        log_lines = []
        log_lines.append("# pKaNET Cloud - Processing Log")
        log_lines.append(f"# Target pH: {target_pH}")
        log_lines.append(f"# Charge mode: {charge_mode}")
        log_lines.append(f"# Stereoisomer enumeration: {'enabled' if enumerate_stereoisomers else 'disabled'}")
        log_lines.append(f"# Total molecules processed: {len(results)}")
        log_lines.append(f"# pKa: {'IUPAC (if matched) → pKaPredict (ML)' if use_iupac_pka else 'pKaPredict (ML)'}")
        log_lines.append("#" + "="*70)
        log_lines.append("")
        log_lines.append("# Columns: Name | pH-adjusted SMILES | Formal Charge | Predicted pKa | pKa Source | Zwitterion")
        log_lines.append("")
        
        for r in results:
            pka_str = f"{r['pka_pred']:.2f}" if r["pka_pred"] is not None else "N/A"
            pka_src = r.get("pka_source", "-")
            zw = "Yes" if r.get("is_zwitterion") else "No"
            log_lines.append(f"{r['name']}\t{r['ph_smiles']}\t{r['formal_charge']:+d}\t{pka_str}\t{pka_src}\t{zw}")
        
        (out / "processing.log").write_text("\n".join(log_lines) + "\n")

    return {"results": results, "summary_text": summary_text, "out_dir": str(out), "format_warnings": format_warnings}

def zip_minimized_structures(out_dir: str, zip_path: str, selected_formats: List[str]) -> str:
    """
    Zip only user-selected structure formats (PDB and/or MOL2), excluding SDF
    
    Args:
        out_dir: Output directory containing structure files
        zip_path: Path for output zip file
        selected_formats: List of user-selected formats (e.g., ["PDB", "MOL2"])
    
    Returns:
        Path to created zip file
    """
    out = Path(out_dir)
    zp = Path(zip_path)
    
    # Convert to lowercase for comparison
    formats_lower = [fmt.lower() for fmt in selected_formats]
    
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.glob("*_min.*"):
            suffix = p.suffix.lower()
            # Only include user-selected formats, exclude .sdf
            if suffix == ".pdb" and "pdb" in formats_lower:
                z.write(p, arcname=p.name)
            elif suffix == ".mol2" and "mol2" in formats_lower:
                z.write(p, arcname=p.name)
    
    return str(zp)


def zip_all_outputs(out_dir: str, zip_path: str) -> str:
    """
    Zip all output files including structures, logs, summaries, and 2D structure PNGs
    """
    out = Path(out_dir)
    zp = Path(zip_path)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(out))
    return str(zp)
