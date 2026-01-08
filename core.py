# core.py
from __future__ import annotations
from pathlib import Path
import os
import zipfile
from typing import Optional, Dict, Any, List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from dimorphite_dl import protonate_smiles
from pkapredict import load_model, smiles_to_rdkit_descriptors, predict_pKa
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


# Load model once (cached in module)
_PKANET_MODEL = None

def get_model():
    global _PKANET_MODEL
    if _PKANET_MODEL is None:
        _PKANET_MODEL = load_model()
    return _PKANET_MODEL


def predict_pka_pkanet(smiles: str) -> float:
    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES for pKa prediction.")
    try:
        desc = smiles_to_rdkit_descriptors([smiles])
    except TypeError:
        desc = smiles_to_rdkit_descriptors([smiles], descriptor_names=None)
    return float(predict_pKa(get_model(), desc)[0])

def ph_adjust_smiles_dimorphite(smiles_str: str, ph: float):
    prot_list = protonate_smiles(smiles_str, ph_min=ph, ph_max=ph, max_variants=1)
    if not prot_list:
        raise ValueError("Dimorphite-DL returned no protonation state.")
    ph_smiles = prot_list[0]
    mol = Chem.MolFromSmiles(ph_smiles)
    if mol is None:
        raise ValueError("RDKit could not parse Dimorphite-DL SMILES.")
    q = Chem.GetFormalCharge(mol)
    return ph_smiles, q

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

def generate_RS_variants(base_smiles: str, base_name: str):
    mol = Chem.MolFromSmiles(base_smiles)
    if mol is None:
        return [{"name": base_name, "stereo": None, "base_smiles": base_smiles}]

    opts = StereoEnumerationOptions(onlyUnassigned=False)
    isomers = list(EnumerateStereoisomers(mol, options=opts))

    if len(isomers) == 1:
        iso_smiles = Chem.MolToSmiles(isomers[0], isomericSmiles=True)
        return [{"name": base_name, "stereo": None, "base_smiles": iso_smiles}]

    iso0 = isomers[0]
    Chem.AssignStereochemistry(iso0, force=True, cleanIt=True)
    centers0 = Chem.FindMolChiralCenters(iso0, includeUnassigned=False)
    if not centers0:
        iso_smiles = Chem.MolToSmiles(isomers[0], isomericSmiles=True)
        return [{"name": base_name, "stereo": None, "base_smiles": iso_smiles}]

    target_idx = centers0[0][0]
    variants = []
    used = set()

    for iso in isomers:
        Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
        centers = Chem.FindMolChiralCenters(iso, includeUnassigned=False)
        label_here = None
        for idx, label in centers:
            if idx == target_idx and label in ("R", "S"):
                label_here = label
                break
        if label_here and label_here not in used:
            used.add(label_here)
            variants.append({
                "name": base_name,
                "stereo": label_here,
                "base_smiles": Chem.MolToSmiles(iso, isomericSmiles=True)
            })
        if used == {"R", "S"}:
            break

    return variants or [{"name": base_name, "stereo": None, "base_smiles": Chem.MolToSmiles(isomers[0], isomericSmiles=True)}]


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
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Default to PDB if no formats specified
    if output_formats is None or len(output_formats) == 0:
        output_formats = ["PDB"]
    
    # User-selected formats (SDF is handled separately, always generated)
    formats_to_save = [fmt.upper() for fmt in output_formats]

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
    if enumerate_stereoisomers:
        for lig in ligands_raw:
            ligands.extend(generate_RS_variants(lig["base_smiles"], lig["name"]))
    else:
        for lig in ligands_raw:
            ligands.append({"name": lig["name"], "stereo": None, "base_smiles": lig["base_smiles"]})

    results = []
    format_warnings = []  # Collect warnings across all molecules
    
    for lig in ligands:
        base_name = lig["name"]
        stereo = lig.get("stereo")
        suffix = f"_{stereo}" if stereo else ""
        pretty_name = base_name + suffix

        base_smiles = lig["base_smiles"]

        try:
            pka_pred = predict_pka_pkanet(base_smiles)
        except Exception:
            pka_pred = None

        ph_smiles, formal_charge = ph_adjust_smiles_dimorphite(base_smiles, target_pH)
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
            "formal_charge": formal_charge,
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
    summary_lines.append(f"Stereoisomer enumeration: {'Enabled' if enumerate_stereoisomers else 'Disabled'}")
    summary_lines.append(f"Total structures generated: {len(results)}")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    for r in results:
        summary_lines.append(f"Molecule: {r['name']}")
        summary_lines.append("-" * 80)
        summary_lines.append(f"  Base SMILES          : {r['base_smiles']}")
        summary_lines.append(f"  pH-adjusted SMILES   : {r['ph_smiles']}")
        
        # Format pKa value safely
        pka_value = f"{r['pka_pred']:.2f}" if r['pka_pred'] is not None else "N/A"
        summary_lines.append(f"  Predicted pKa        : {pka_value}")
        summary_lines.append(f"  Formal Charge (pH {target_pH}): {r['formal_charge']:+d}")
        
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
    summary_lines.append("pKa Prediction powered by pKaPredict (ML-based)")
    summary_lines.append("=" * 80)
    
    summary_text = "\n".join(summary_lines).strip()
    (out / "summary.txt").write_text(summary_text + "\n")
    
    # Create log file for SMI_FILE input
    if input_type == "SMI_FILE" and len(results) > 0:
        log_lines = []
        log_lines.append("# pKaNET Cloud - Processing Log")
        log_lines.append(f"# Target pH: {target_pH}")
        log_lines.append(f"# Stereoisomer enumeration: {'enabled' if enumerate_stereoisomers else 'disabled'}")
        log_lines.append(f"# Total molecules processed: {len(results)}")
        log_lines.append(f"# pKa prediction: pKaPredict (Machine Learning)")
        log_lines.append("#" + "="*70)
        log_lines.append("")
        log_lines.append("# Columns: Name | pH-adjusted SMILES | Formal Charge | Predicted pKa")
        log_lines.append("")
        
        for r in results:
            pka_str = f"{r['pka_pred']:.2f}" if r["pka_pred"] is not None else "N/A"
            log_lines.append(f"{r['name']}\t{r['ph_smiles']}\t{r['formal_charge']:+d}\t{pka_str}")
        
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
