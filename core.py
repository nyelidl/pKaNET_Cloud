# core.py
"""
pKaNET Cloud - Core Processing Module
Refined for Streamlit integration with improved error handling and progress tracking
"""

from __future__ import annotations
from pathlib import Path
import os
import zipfile
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from dimorphite_dl import protonate_smiles
from pkapredict import load_model, predict_pKa
import subprocess
import shutil


# ============================================================================
# Data Classes for Better Type Safety
# ============================================================================

@dataclass
class ProcessingResult:
    """Result from processing a single molecule"""
    name: str
    base_smiles: str
    ph_smiles: str
    pka_pred: Optional[float]
    formal_charge: int
    stereoisomer_id: Optional[str] = None
    minimized_pdb: Optional[str] = None
    minimized_sdf: Optional[str] = None
    minimized_mol2: Optional[str] = None
    png_2d: Optional[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class JobResult:
    """Complete job result with all processed molecules"""
    results: List[ProcessingResult]
    summary_text: str
    out_dir: str
    format_warnings: List[str]
    total_molecules: int
    successful: int
    failed: int


# ============================================================================
# Open Babel Support
# ============================================================================

_OBABEL_AVAILABLE = None

def check_obabel() -> bool:
    """Check if obabel command is available"""
    global _OBABEL_AVAILABLE
    if _OBABEL_AVAILABLE is None:
        _OBABEL_AVAILABLE = shutil.which("obabel") is not None
    return _OBABEL_AVAILABLE


def convert_pdb_to_mol2_obabel(pdb_path: str, mol2_path: str) -> Tuple[bool, Optional[str]]:
    """
    Convert PDB to MOL2 using Open Babel
    
    Args:
        pdb_path: Path to input PDB file
        mol2_path: Path to output MOL2 file
    
    Returns:
        Tuple of (success, error_message)
    """
    if not check_obabel():
        return False, "Open Babel (obabel) not found in PATH"
    
    try:
        result = subprocess.run(
            ["obabel", pdb_path, "-O", mol2_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and Path(mol2_path).exists():
            return True, None
        else:
            return False, f"Conversion failed: {result.stderr[:200]}"
            
    except subprocess.TimeoutExpired:
        return False, "Conversion timed out (>30s)"
    except Exception as e:
        return False, f"Conversion error: {str(e)}"


# ============================================================================
# pKa Prediction Model
# ============================================================================

_PKANET_MODEL = None
_DESCRIPTOR_NAMES = None

def get_model() -> Tuple[Any, List[str]]:
    """Load and cache the pKa prediction model"""
    global _PKANET_MODEL, _DESCRIPTOR_NAMES
    if _PKANET_MODEL is None:
        _PKANET_MODEL = load_model()
        
        if hasattr(_PKANET_MODEL, 'feature_name_'):
            _DESCRIPTOR_NAMES = _PKANET_MODEL.feature_name_
        else:
            from rdkit.Chem import Descriptors
            all_descriptors = [desc[0] for desc in Descriptors._descList]
            _DESCRIPTOR_NAMES = all_descriptors[:_PKANET_MODEL.n_features_]
    
    return _PKANET_MODEL, _DESCRIPTOR_NAMES


def predict_pka_pkanet(smiles: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Predict pKa using pKaPredict ML model
    
    Args:
        smiles: SMILES string
    
    Returns:
        Tuple of (pka_value, error_message)
    """
    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES for pKa prediction"
    
    try:
        model, descriptor_names = get_model()
        pka_value = predict_pKa(smiles, model, descriptor_names)
        
        # Handle various return types
        if isinstance(pka_value, (list, tuple)):
            pka_value = pka_value[0]
        elif hasattr(pka_value, '__iter__') and not isinstance(pka_value, str):
            pka_value = next(iter(pka_value))
            
        return float(pka_value), None
        
    except Exception as e:
        return None, f"pKa prediction failed: {str(e)[:100]}"


# ============================================================================
# Chemistry Utilities
# ============================================================================

def _has_acidic_group(mol: Chem.Mol) -> bool:
    """Check if molecule has common acidic groups"""
    acid_smarts = [
        "C(=O)[O;H1]",          # COOH
        "S(=O)(=O)[O;H1]",      # SO3H
        "P(=O)(O)(O)O",         # phosphoric acid
    ]
    return any(mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in acid_smarts)


def ph_adjust_smiles_dimorphite(smiles_str: str, ph: float) -> Tuple[str, int, Optional[str]]:
    """
    Adjust SMILES protonation state for target pH using Dimorphite-DL
    
    Args:
        smiles_str: Input SMILES
        ph: Target pH
    
    Returns:
        Tuple of (ph_adjusted_smiles, formal_charge, error_message)
    """
    try:
        prot_list = protonate_smiles(smiles_str, ph_min=ph, ph_max=ph, max_variants=32)
        if not prot_list:
            return smiles_str, 0, "Dimorphite-DL returned no protonation states"

        candidates = []
        for smi in prot_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            q = Chem.GetFormalCharge(mol)
            candidates.append((smi, q, mol))

        if not candidates:
            return smiles_str, 0, "No valid protonation states generated"

        # Filter unrealistic negative charges
        filtered = []
        for smi, q, mol in candidates:
            if (q < 0) and (not _has_acidic_group(mol)):
                continue
            filtered.append((smi, q, mol))

        if not filtered:
            filtered = candidates

        # Prefer neutral charge, then smallest absolute charge
        filtered.sort(key=lambda x: (abs(x[1]),))
        ph_smiles, q, _ = filtered[0]
        return ph_smiles, q, None
        
    except Exception as e:
        return smiles_str, 0, f"pH adjustment failed: {str(e)[:100]}"


def build_minimized_3d(smiles: str) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """
    Build and minimize 3D structure from SMILES
    
    Args:
        smiles: Input SMILES string
    
    Returns:
        Tuple of (minimized_molecule, error_message)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES for 3D generation"
        
        mol = Chem.AddHs(mol)

        # Try ETKDG embedding
        try:
            params = AllChem.ETKDGv3() if hasattr(AllChem, 'ETKDGv3') else AllChem.ETKDG()
            params.randomSeed = 0xF00D
            code = AllChem.EmbedMolecule(mol, params)
        except Exception:
            code = AllChem.EmbedMolecule(mol, randomSeed=0xF00D, maxAttempts=2000)

        # Fallback to random coords if needed
        if code != 0 or mol.GetNumConformers() == 0:
            code2 = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=0xF00D, maxAttempts=2000)
            if code2 != 0 or mol.GetNumConformers() == 0:
                return None, "3D embedding failed - no conformer generated"

        # Force field optimization
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception as e:
            # Continue even if optimization fails
            pass
            
        return mol, None
        
    except Exception as e:
        return None, f"3D generation error: {str(e)[:100]}"


# ============================================================================
# File I/O
# ============================================================================

def parse_smi_lines(text: str) -> List[Tuple[str, str]]:
    """Parse SMI file format (SMILES name pairs)"""
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


def generate_RS_variants(base_smiles: str, base_name: str) -> List[Dict[str, Any]]:
    """
    Generate R/S stereoisomer variants
    
    Args:
        base_smiles: Input SMILES
        base_name: Base molecule name
    
    Returns:
        List of variant dictionaries with name, stereo, and base_smiles
    """
    mol = Chem.MolFromSmiles(base_smiles)
    if mol is None:
        return [{"name": base_name, "stereo": None, "base_smiles": base_smiles}]

    opts = StereoEnumerationOptions(onlyUnassigned=False)
    try:
        isomers = list(EnumerateStereoisomers(mol, options=opts))
    except Exception:
        return [{"name": base_name, "stereo": None, "base_smiles": base_smiles}]

    if len(isomers) == 1:
        iso_smiles = Chem.MolToSmiles(isomers[0], isomericSmiles=True)
        return [{"name": base_name, "stereo": None, "base_smiles": iso_smiles}]

    # Find chiral centers
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


def save_2d_structure_image(smiles: str, output_path: str, size=(800, 600)) -> Tuple[bool, Optional[str]]:
    """
    Save 2D structure as PNG image
    
    Args:
        smiles: SMILES string
        output_path: Path to save PNG file
        size: Image size (width, height)
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        from rdkit.Chem import Draw
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES for 2D image"
        
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=size)
        img.save(output_path)
        return True, None
        
    except ImportError:
        return False, "RDKit drawing module not available"
    except Exception as e:
        return False, f"Image generation failed: {str(e)[:100]}"


def save_molecule_files(mol: Chem.Mol, base_path: str, formats: List[str]) -> Dict[str, Any]:
    """
    Save molecule to multiple file formats
    Always generates SDF for visualization. User-selected formats are also saved.
    
    Args:
        mol: RDKit molecule object
        base_path: Base file path without extension
        formats: List of formats to save (e.g., ["PDB", "MOL2"])
    
    Returns:
        Dictionary with 'files' and 'warnings'
    """
    saved_files = {}
    warnings = []
    mol2_via_obabel = False
    
    # Always save SDF first (for visualization)
    try:
        sdf_path = f"{base_path}.sdf"
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        saved_files["sdf"] = sdf_path
    except Exception as e:
        warnings.append(f"⚠️ SDF generation failed: {str(e)[:100]}")
    
    # Save user-requested formats
    for fmt in formats:
        fmt_upper = fmt.upper()
        
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
                    except Exception:
                        pass
                
                # Fallback to Open Babel
                if "pdb" not in saved_files:
                    pdb_path = f"{base_path}.pdb"
                    try:
                        Chem.MolToPDBFile(mol, pdb_path)
                        saved_files["pdb"] = pdb_path
                    except Exception as e:
                        warnings.append(f"⚠️ PDB generation for MOL2 conversion failed")
                        continue
                
                pdb_path = saved_files.get("pdb")
                if pdb_path:
                    success, error = convert_pdb_to_mol2_obabel(pdb_path, file_path)
                    if success:
                        saved_files["mol2"] = file_path
                        mol2_via_obabel = True
                    else:
                        if not check_obabel():
                            warnings.append("⚠️ MOL2 format requires Open Babel (obabel)")
                        else:
                            warnings.append(f"⚠️ MOL2 conversion failed: {error}")
        
        except Exception as e:
            warnings.append(f"⚠️ {fmt_upper} generation failed: {str(e)[:100]}")
    
    if mol2_via_obabel:
        warnings.append("ℹ️ MOL2 files generated via Open Babel")
    
    return {"files": saved_files, "warnings": warnings}


# ============================================================================
# Main Processing Pipeline
# ============================================================================

def run_job(
    *,
    input_type: str,
    smiles_text: Optional[str] = None,
    uploaded_bytes: Optional[bytes] = None,
    uploaded_name: Optional[str] = None,
    target_pH: float = 7.4,
    output_name: str = "ligand",
    out_dir: str = "./output",
    output_formats: Optional[List[str]] = None,
    enumerate_stereoisomers: bool = True,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> JobResult:
    """
    Main processing pipeline for pKaNET Cloud
    
    Args:
        input_type: "SMILES", "SMI_FILE", or "FILE"
        smiles_text: SMILES string (for SMILES input)
        uploaded_bytes: File content (for file inputs)
        uploaded_name: Original filename
        target_pH: Target pH for protonation
        output_name: Output file base name
        out_dir: Output directory
        output_formats: List of output formats (default: ["PDB"])
        enumerate_stereoisomers: Whether to enumerate R/S variants
        progress_callback: Optional callback(message, progress) for UI updates
    
    Returns:
        JobResult object with processing results
    """
    def update_progress(msg: str, pct: float):
        if progress_callback:
            progress_callback(msg, pct)
    
    update_progress("Initializing...", 0.0)
    
    # Setup output directory
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Default to PDB if no formats specified
    if output_formats is None or len(output_formats) == 0:
        output_formats = ["PDB"]
    
    formats_to_save = [fmt.upper() for fmt in output_formats]

    # ========================================================================
    # Parse Input
    # ========================================================================
    
    ligands_raw = []
    
    try:
        if input_type == "SMILES":
            update_progress("Parsing SMILES...", 0.1)
            base_smiles = (smiles_text or "").strip()
            if not base_smiles:
                raise ValueError("SMILES input is empty")
            ligands_raw.append({"name": output_name, "base_smiles": base_smiles})

        elif input_type == "SMI_FILE":
            update_progress("Parsing SMI file...", 0.1)
            if not uploaded_bytes:
                raise ValueError("No .smi file uploaded")
            text = uploaded_bytes.decode("utf-8", errors="replace")
            parsed = parse_smi_lines(text)
            if not parsed:
                raise ValueError("No valid SMILES found in .smi file")
            for smi, name in parsed:
                ligands_raw.append({"name": name, "base_smiles": smi})

        elif input_type == "FILE":
            update_progress("Reading uploaded file...", 0.1)
            if not uploaded_bytes or not uploaded_name:
                raise ValueError("No ligand file uploaded")
            
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
                raise ValueError(f"Unsupported file type: {ext}. Use .pdb, .mol2, or .sdf")

            if mol_in is None:
                raise ValueError("RDKit could not parse uploaded file")

            # Clean up multi-fragment molecules
            try:
                frags = Chem.GetMolFrags(mol_in, asMols=True, sanitizeFrags=False)
                if len(frags) > 1:
                    mol_in = max(frags, key=lambda m: m.GetNumHeavyAtoms())
                Chem.SanitizeMol(mol_in)
            except Exception:
                pass

            base_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol_in), canonical=True)
            ligands_raw.append({
                "name": output_name or os.path.splitext(uploaded_name)[0], 
                "base_smiles": base_smiles
            })

        else:
            raise ValueError(f"Unknown input_type: {input_type}")
            
    except Exception as e:
        raise ValueError(f"Input parsing failed: {str(e)}")

    # ========================================================================
    # Enumerate Stereoisomers
    # ========================================================================
    
    update_progress("Enumerating stereoisomers...", 0.2)
    ligands = []
    if enumerate_stereoisomers:
        for lig in ligands_raw:
            ligands.extend(generate_RS_variants(lig["base_smiles"], lig["name"]))
    else:
        for lig in ligands_raw:
            ligands.append({"name": lig["name"], "stereo": None, "base_smiles": lig["base_smiles"]})

    # ========================================================================
    # Process Each Molecule
    # ========================================================================
    
    results = []
    format_warnings = set()
    successful = 0
    failed = 0
    
    total = len(ligands)
    for i, lig in enumerate(ligands):
        base_name = lig["name"]
        stereo = lig.get("stereo")
        suffix = f"_{stereo}" if stereo else ""
        pretty_name = base_name + suffix
        
        progress = 0.2 + (0.7 * (i / total))
        update_progress(f"Processing {pretty_name}...", progress)

        base_smiles = lig["base_smiles"]
        errors = []

        # Predict pKa
        pka_pred, pka_error = predict_pka_pkanet(base_smiles)
        if pka_error:
            errors.append(pka_error)

        # pH adjustment
        ph_smiles, formal_charge, ph_error = ph_adjust_smiles_dimorphite(base_smiles, target_pH)
        if ph_error:
            errors.append(ph_error)

        # Build 3D structure
        mol_min, build_error = build_minimized_3d(ph_smiles)
        if build_error:
            errors.append(build_error)
            failed += 1
            results.append(ProcessingResult(
                name=pretty_name,
                base_smiles=base_smiles,
                ph_smiles=ph_smiles,
                pka_pred=pka_pred,
                formal_charge=formal_charge,
                stereoisomer_id=stereo,
                errors=errors
            ))
            continue

        # Save molecule files
        base_file_path = str(out / f"{base_name}{suffix}_min")
        save_result = save_molecule_files(mol_min, base_file_path, formats_to_save)
        saved_files = save_result["files"]
        
        # Collect warnings
        for warning in save_result["warnings"]:
            format_warnings.add(warning)
        
        # Save 2D structure PNG
        png_path = str(out / f"{base_name}{suffix}_2D.png")
        png_success, png_error = save_2d_structure_image(ph_smiles, png_path)
        if png_error:
            errors.append(f"2D image: {png_error}")

        # Create result
        result = ProcessingResult(
            name=pretty_name,
            base_smiles=base_smiles,
            ph_smiles=ph_smiles,
            pka_pred=pka_pred,
            formal_charge=formal_charge,
            stereoisomer_id=stereo,
            minimized_pdb=saved_files.get("pdb"),
            minimized_sdf=saved_files.get("sdf"),
            minimized_mol2=saved_files.get("mol2"),
            png_2d=png_path if png_success else None,
            errors=errors
        )
        
        results.append(result)
        if not errors or (len(errors) == 1 and "pKa" in errors[0]):
            successful += 1
        else:
            failed += 1

    # ========================================================================
    # Generate Summary
    # ========================================================================
    
    update_progress("Generating summary...", 0.95)
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("pKaNET Cloud - Analysis Summary")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Target pH: {target_pH}")
    summary_lines.append(f"Stereoisomer enumeration: {'Enabled' if enumerate_stereoisomers else 'Disabled'}")
    summary_lines.append(f"Total structures: {len(results)} ({successful} successful, {failed} failed)")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    for r in results:
        summary_lines.append(f"Molecule: {r.name}")
        summary_lines.append("-" * 80)
        summary_lines.append(f"  Base SMILES          : {r.base_smiles}")
        summary_lines.append(f"  pH-adjusted SMILES   : {r.ph_smiles}")
        
        pka_value = f"{r.pka_pred:.2f}" if r.pka_pred is not None else "N/A"
        summary_lines.append(f"  Predicted pKa        : {pka_value}")
        summary_lines.append(f"  Formal Charge (pH {target_pH}): {r.formal_charge:+d}")
        
        generated_formats = []
        if r.minimized_pdb:
            generated_formats.append("PDB")
        if r.minimized_mol2:
            generated_formats.append("MOL2")
        if r.minimized_sdf:
            generated_formats.append("SDF")
        summary_lines.append(f"  Output Formats       : {', '.join(generated_formats) if generated_formats else 'None'}")
        
        if r.stereoisomer_id:
            summary_lines.append(f"  Stereoisomer         : {r.stereoisomer_id}")
            
        if r.errors:
            summary_lines.append(f"  Errors/Warnings      : {'; '.join(r.errors)}")
        
        summary_lines.append("")
    
    summary_lines.append("=" * 80)
    summary_lines.append("pKa Prediction: pKaPredict (Machine Learning)")
    summary_lines.append("=" * 80)
    
    summary_text = "\n".join(summary_lines)
    (out / "summary.txt").write_text(summary_text + "\n")
    
    # Generate processing log for SMI_FILE input
    if input_type == "SMI_FILE" and results:
        log_lines = []
        log_lines.append("# pKaNET Cloud - Processing Log")
        log_lines.append(f"# Target pH: {target_pH}")
        log_lines.append(f"# Stereoisomers: {'enabled' if enumerate_stereoisomers else 'disabled'}")
        log_lines.append(f"# Total: {len(results)} ({successful} successful, {failed} failed)")
        log_lines.append("#" + "="*70)
        log_lines.append("# Name | pH-SMILES | Charge | pKa | Status")
        log_lines.append("")
        
        for r in results:
            pka_str = f"{r.pka_pred:.2f}" if r.pka_pred is not None else "N/A"
            status = "OK" if not r.errors else "WARNING"
            log_lines.append(f"{r.name}\t{r.ph_smiles}\t{r.formal_charge:+d}\t{pka_str}\t{status}")
        
        (out / "processing.log").write_text("\n".join(log_lines) + "\n")

    update_progress("Complete!", 1.0)
    
    return JobResult(
        results=results,
        summary_text=summary_text,
        out_dir=str(out),
        format_warnings=sorted(list(format_warnings)),
        total_molecules=len(results),
        successful=successful,
        failed=failed
    )


# ============================================================================
# ZIP Utilities
# ============================================================================

def zip_minimized_structures(out_dir: str, zip_path: str, selected_formats: List[str]) -> str:
    """
    Zip only user-selected structure formats (excludes SDF)
    
    Args:
        out_dir: Output directory containing structure files
        zip_path: Path for output zip file
        selected_formats: User-selected formats (e.g., ["PDB", "MOL2"])
    
    Returns:
        Path to created zip file
    """
    out = Path(out_dir)
    zp = Path(zip_path)
    
    formats_lower = [fmt.lower() for fmt in selected_formats]
    
    file_count = 0
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.glob("*_min.*"):
            suffix = p.suffix.lower()
            if suffix == ".pdb" and "pdb" in formats_lower:
                z.write(p, arcname=p.name)
                file_count += 1
            elif suffix == ".mol2" and "mol2" in formats_lower:
                z.write(p, arcname=p.name)
                file_count += 1
    
    if file_count == 0:
        raise ValueError("No structure files found to zip")
    
    return str(zp)


def zip_all_outputs(out_dir: str, zip_path: str) -> str:
    """
    Zip all output files including structures, logs, summaries, and 2D images
    
    Args:
        out_dir: Output directory
        zip_path: Path for output zip file
    
    Returns:
        Path to created zip file
    """
    out = Path(out_dir)
    zp = Path(zip_path)
    
    file_count = 0
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(out))
                file_count += 1
    
    if file_count == 0:
        raise ValueError("No files found to zip")
    
    return str(zp)
