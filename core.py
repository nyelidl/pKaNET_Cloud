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


# Load model once (cached in module)
_PKANET_MODEL = None

def get_model():
    global _PKANET_MODEL
    if _PKANET_MODEL is None:
        _PKANET_MODEL = load_model()
    return _PKANET_MODEL


# ---- paste your helper functions here (edited: remove IPython/display/py3Dmol) ----
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


def save_molecule_files(mol, base_path: str, formats: List[str]) -> Dict[str, str]:
    """
    Save molecule to multiple file formats.
    
    Args:
        mol: RDKit molecule object
        base_path: Base file path without extension
        formats: List of formats to save (e.g., ["PDB", "SDF", "MOL2"])
    
    Returns:
        Dictionary mapping format to file path
    """
    saved_files = {}
    
    for fmt in formats:
        fmt_upper = fmt.upper()
        
        try:
            if fmt_upper == "PDB":
                file_path = f"{base_path}.pdb"
                Chem.MolToPDBFile(mol, file_path)
                saved_files["pdb"] = file_path
                
            elif fmt_upper == "SDF":
                file_path = f"{base_path}.sdf"
                writer = Chem.SDWriter(file_path)
                writer.write(mol)
                writer.close()
                saved_files["sdf"] = file_path
                
            elif fmt_upper == "MOL2":
                file_path = f"{base_path}.mol2"
                # Try to write MOL2, fall back to SDF if not available
                try:
                    Chem.MolToMol2File(mol, file_path)
                    saved_files["mol2"] = file_path
                except (AttributeError, RuntimeError) as e:
                    # MOL2 writing not available in this RDKit build
                    print(f"Warning: MOL2 format not available, using SDF instead. Error: {e}")
                    file_path = f"{base_path}.sdf"
                    if "sdf" not in saved_files:
                        writer = Chem.SDWriter(file_path)
                        writer.write(mol)
                        writer.close()
                        saved_files["mol2"] = file_path  # Map mol2 request to sdf file
                    else:
                        saved_files["mol2"] = saved_files["sdf"]
        
        except Exception as e:
            print(f"Warning: Could not save {fmt_upper} format: {e}")
            continue
    
    return saved_files


def run_job(
    *,
    input_type: str,          # "SMILES" | "SMI_FILE" | "FILE"
    smiles_text: str | None,
    uploaded_bytes: bytes | None,
    uploaded_name: str | None,
    target_pH: float,
    output_name: str,
    out_dir: str,
    output_formats: List[str] = None,  # ["PDB", "MOL2"] - SDF is always generated
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Default to PDB if no formats specified
    if output_formats is None or len(output_formats) == 0:
        output_formats = ["PDB"]
    
    # Always include SDF for 3D visualization (even if not in user selection)
    formats_to_save = list(output_formats)
    if "SDF" not in formats_to_save:
        formats_to_save.append("SDF")

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

    ligands = []
    for lig in ligands_raw:
        ligands.extend(generate_RS_variants(lig["base_smiles"], lig["name"]))

    results = []
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

        # Save in requested formats (plus SDF for visualization)
        base_file_path = str(out / f"{base_name}{suffix}_min")
        saved_files = save_molecule_files(mol_min, base_file_path, formats_to_save)

        result_entry = {
            "name": pretty_name,
            "base_smiles": base_smiles,
            "ph_smiles": ph_smiles,
            "pka_pred": pka_pred,
            "formal_charge": formal_charge,
        }
        
        # Add file paths to result (SDF always included for visualization)
        if "pdb" in saved_files:
            result_entry["minimized_pdb"] = saved_files["pdb"]
        if "sdf" in saved_files:
            result_entry["minimized_sdf"] = saved_files["sdf"]
        if "mol2" in saved_files:
            result_entry["minimized_mol2"] = saved_files["mol2"]
        
        results.append(result_entry)

    # write a summary file for the UI
    summary_lines = []
    for r in results:
        summary_lines.append(f"{r['name']}")
        summary_lines.append(f"  Base SMILES: {r['base_smiles']}")
        summary_lines.append(f"  pH SMILES  : {r['ph_smiles']}")
        if r["pka_pred"] is not None:
            summary_lines.append(f"  pKa (ML)   : {r['pka_pred']:.2f}")
        summary_lines.append(f"  Charge     : {r['formal_charge']}")
        summary_lines.append(f"  Formats    : {', '.join(output_formats)}")
        summary_lines.append("")
    summary_text = "\n".join(summary_lines).strip()
    (out / "summary.txt").write_text(summary_text + "\n")
    
    # Create log file for SMI_FILE input with SMILES and charges
    if input_type == "SMI_FILE" and len(results) > 0:
        log_lines = []
        log_lines.append("# pKaNET Cloud - Processing Log")
        log_lines.append(f"# Target pH: {target_pH}")
        log_lines.append(f"# Total molecules processed: {len(results)}")
        log_lines.append("#" + "="*70)
        log_lines.append("")
        log_lines.append("# Format: Name | pH-adjusted SMILES | Formal Charge | pKa (predicted)")
        log_lines.append("")
        
        for r in results:
            pka_str = f"{r['pka_pred']:.2f}" if r["pka_pred"] is not None else "N/A"
            log_lines.append(f"{r['name']}\t{r['ph_smiles']}\t{r['formal_charge']}\t{pka_str}")
        
        (out / "processing.log").write_text("\n".join(log_lines) + "\n")

    return {"results": results, "summary_text": summary_text, "out_dir": str(out)}

def zip_minimized_pdb_only(out_dir: str, zip_path: str) -> str:
    """Zip all minimized structure files (PDB, SDF, MOL2)"""
    out = Path(out_dir)
    zp = Path(zip_path)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.glob("*_min.*"):
            if p.suffix.lower() in [".pdb", ".sdf", ".mol2"]:
                z.write(p, arcname=p.name)
    return str(zp)

def zip_all_outputs(out_dir: str, zip_path: str) -> str:
    out = Path(out_dir)
    zp = Path(zip_path)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(out))
    return str(zp)
