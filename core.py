# core.py
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolfiles
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from dimorphite_dl import protonate_smiles
from pkapredict import load_model, smiles_to_rdkit_descriptors, predict_pKa


# -----------------------------
# Model cache (loaded once)
# -----------------------------
_PKANET_MODEL = None


def get_pkanet_model():
    global _PKANET_MODEL
    if _PKANET_MODEL is None:
        _PKANET_MODEL = load_model()
    return _PKANET_MODEL


# -----------------------------
# Core helpers
# -----------------------------
def predict_pka_pkanet(smiles: str) -> float:
    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES for pKa prediction.")

    try:
        desc = smiles_to_rdkit_descriptors([smiles])
    except TypeError:
        # some versions require descriptor_names
        desc = smiles_to_rdkit_descriptors([smiles], descriptor_names=None)

    y = predict_pKa(get_pkanet_model(), desc)
    return float(y[0])


def ph_adjust_smiles_dimorphite(smiles_str: str, ph: float) -> Tuple[str, int]:
    # returns a single best variant at the given pH
    prot_list = protonate_smiles(smiles_str, ph_min=ph, ph_max=ph, max_variants=1)
    if not prot_list:
        raise ValueError("Dimorphite-DL returned no protonation state.")
    ph_smiles = prot_list[0]

    mol = Chem.MolFromSmiles(ph_smiles)
    if mol is None:
        raise ValueError("RDKit could not parse pH-adjusted SMILES.")
    charge = Chem.GetFormalCharge(mol)
    return ph_smiles, charge


def build_minimized_3d(smiles: str) -> Chem.Mol:
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
        # fallback embedding
        code2 = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=0xF00D, maxAttempts=2000)
        if code2 != 0 or mol.GetNumConformers() == 0:
            raise ValueError("3D embedding failed (no conformer).")

    # minimize (MMFF if possible)
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        # still usable even if minimization fails
        pass

    return mol


def parse_smi_lines(text: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
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
    mol = Chem.MolFromSmiles(base_smiles)
    if mol is None:
        return [{"name": base_name, "stereo": None, "base_smiles": base_smiles}]

    opts = StereoEnumerationOptions(onlyUnassigned=False)
    isomers = list(EnumerateStereoisomers(mol, options=opts))

    if len(isomers) == 1:
        iso_smiles = Chem.MolToSmiles(isomers[0], isomericSmiles=True)
        return [{"name": base_name, "stereo": None, "base_smiles": iso_smiles}]

    # pick one chiral center to label R/S variants (simple, consistent behavior)
    iso0 = isomers[0]
    Chem.AssignStereochemistry(iso0, force=True, cleanIt=True)
    centers0 = Chem.FindMolChiralCenters(iso0, includeUnassigned=False)
    if not centers0:
        iso_smiles = Chem.MolToSmiles(isomers[0], isomericSmiles=True)
        return [{"name": base_name, "stereo": None, "base_smiles": iso_smiles}]

    target_idx = centers0[0][0]  # first chiral center
    variants: List[Dict[str, Any]] = []
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

    if not variants:
        variants = [{"name": base_name, "stereo": None, "base_smiles": Chem.MolToSmiles(isomers[0], isomericSmiles=True)}]
    return variants


def _extract_smiles_from_uploaded_ligand(uploaded_bytes: bytes, uploaded_name: str, out_dir: Path) -> str:
    ext = os.path.splitext(uploaded_name)[1].lower()
    tmp_path = out_dir / f"uploaded{ext}"
    tmp_path.write_bytes(uploaded_bytes)

    mol_in: Optional[Chem.Mol] = None
    if ext == ".pdb":
        mol_in = Chem.MolFromPDBFile(str(tmp_path), removeHs=False, sanitize=False)
    elif ext == ".mol2":
        mol_in = Chem.MolFromMol2File(str(tmp_path), removeHs=False, sanitize=False)
    elif ext == ".sdf":
        supplier = Chem.SDMolSupplier(str(tmp_path), removeHs=False, sanitize=False)
        mol_in = next((m for m in supplier if m is not None), None)
    else:
        raise ValueError("Unsupported file type. Use .pdb, .mol2, or .sdf.")

    if mol_in is None:
        raise ValueError("RDKit could not parse the uploaded ligand.")

    # attempt sanitize and choose largest fragment if multiple
    try:
        frags = Chem.GetMolFrags(mol_in, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            mol_in = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        Chem.SanitizeMol(mol_in)
    except Exception:
        pass

    base_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol_in), canonical=True)
    if not base_smiles:
        raise ValueError("Failed to derive SMILES from uploaded ligand.")
    return base_smiles


def _write_output(mol: Chem.Mol, base_path_noext: Path, output_format: str) -> Tuple[str, Optional[str]]:
    """
    Writes the minimized structure in requested format.
    Returns: (written_file_path, warning_message_or_None)
    """
    fmt = output_format.upper().strip()
    warning = None

    if fmt == "PDB":
        out_path = base_path_noext.with_suffix(".pdb")
        pdb_block = Chem.MolToPDBBlock(mol)
        out_path.write_text(pdb_block)
        return str(out_path), warning

    if fmt == "SDF":
        out_path = base_path_noext.with_suffix(".sdf")
        w = Chem.SDWriter(str(out_path))
        w.write(mol)
        w.close()
        return str(out_path), warning

    if fmt == "MOL2":
        out_path = base_path_noext.with_suffix(".mol2")
        try:
            # Mol2 writer availability depends on RDKit build
            rdmolfiles.MolToMol2File(mol, str(out_path))
            return str(out_path), warning
        except Exception:
            # graceful fallback to SDF (Streamlit Cloud often lacks MOL2 writer support)
            warning = "MOL2 writing is not available on this server build; saved as SDF instead."
            out_path = base_path_noext.with_suffix(".sdf")
            w = Chem.SDWriter(str(out_path))
            w.write(mol)
            w.close()
            return str(out_path), warning

    raise ValueError("Unknown output format. Choose PDB, SDF, or MOL2.")


# -----------------------------
# Main job runner
# -----------------------------
def run_job(
    *,
    input_type: str,  # "SMILES" | "SMI_FILE" | "FILE"
    smiles_text: Optional[str],
    uploaded_bytes: Optional[bytes],
    uploaded_name: Optional[str],
    target_pH: float,
    output_name: str,
    output_format: str,  # "PDB" | "SDF" | "MOL2"
    out_dir: str,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Build ligand list from input
    ligands_raw: List[Dict[str, str]] = []

    if input_type == "SMILES":
        base_smiles = (smiles_text or "").strip()
        if not base_smiles:
            raise ValueError("SMILES is empty.")
        ligands_raw.append({"name": output_name or "ligand", "base_smiles": base_smiles})

    elif input_type == "SMI_FILE":
        if not uploaded_bytes:
            raise ValueError("No .smi file uploaded.")
        text = uploaded_bytes.decode("utf-8", errors="replace")
        for smi, name in parse_smi_lines(text):
            ligands_raw.append({"name": name, "base_smiles": smi})

    elif input_type == "FILE":
        if not uploaded_bytes or not uploaded_name:
            raise ValueError("No ligand file uploaded.")
        base_smiles = _extract_smiles_from_uploaded_ligand(uploaded_bytes, uploaded_name, out)
        ligands_raw.append({"name": output_name or Path(uploaded_name).stem, "base_smiles": base_smiles})

    else:
        raise ValueError("Unknown input_type. Use SMILES, SMI_FILE, or FILE.")

    # 2) Enumerate R/S variants (simple R/S for the first center)
    ligands: List[Dict[str, Any]] = []
    for lig in ligands_raw:
        ligands.extend(generate_RS_variants(lig["base_smiles"], lig["name"]))

    # 3) Process each ligand: pKa → protonation → 3D → minimize → write file(s)
    results: List[Dict[str, Any]] = []
    summary_lines: List[str] = []

    for lig in ligands:
        base_name = lig["name"]
        stereo = lig.get("stereo")
        suffix = f"_{stereo}" if stereo else ""
        pretty_name = base_name + suffix

        base_smiles = str(lig["base_smiles"])

        # pKa prediction (non-fatal)
        try:
            pka_pred = predict_pka_pkanet(base_smiles)
        except Exception:
            pka_pred = None

        ph_smiles, formal_charge = ph_adjust_smiles_dimorphite(base_smiles, target_pH)
        mol_min = build_minimized_3d(ph_smiles)

        pdb_block = Chem.MolToPDBBlock(mol_min)  # always for viewer

        base_out = out / f"{base_name}{suffix}_min"
        written_path, warning = _write_output(mol_min, base_out, output_format)

        # also always save a PDB copy for debugging/viewer consistency
        viewer_pdb_path = out / f"{base_name}{suffix}_min_viewer.pdb"
        viewer_pdb_path.write_text(pdb_block)

        item: Dict[str, Any] = {
            "name": pretty_name,
            "base_smiles": base_smiles,
            "ph_smiles": ph_smiles,
            "pka_pred": pka_pred,
            "formal_charge": formal_charge,
            "output_file": written_path,
            "viewer_pdb_file": str(viewer_pdb_path),
            "pdb_text_for_viewer": pdb_block,
        }
        if warning:
            item["warning"] = warning

        results.append(item)

        # summary
        summary_lines.append(f"{pretty_name}")
        summary_lines.append(f"  Base SMILES: {base_smiles}")
        summary_lines.append(f"  pH SMILES  : {ph_smiles}")
        if pka_pred is not None:
            summary_lines.append(f"  pKa (ML)   : {pka_pred:.2f}")
        summary_lines.append(f"  Charge     : {formal_charge}")
        summary_lines.append(f"  Output     : {Path(written_path).name}")
        if warning:
            summary_lines.append(f"  Note       : {warning}")
        summary_lines.append("")

    summary_text = "\n".join(summary_lines).strip() + "\n"
    (out / "summary.txt").write_text(summary_text)

    return {"results": results, "summary_text": summary_text, "out_dir": str(out)}


# -----------------------------
# Zipping outputs
# -----------------------------
def zip_all_outputs(out_dir: str, zip_path: str) -> str:
    out = Path(out_dir)
    zp = Path(zip_path)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(out))
    return str(zp)


def zip_minimized_only(out_dir: str, zip_path: str) -> str:
    """
    Zip only the primary output files (*_min.<ext>) + summary.txt
    """
    out = Path(out_dir)
    zp = Path(zip_path)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.glob("*_min.*"):
            if p.is_file():
                z.write(p, arcname=p.name)
        s = out / "summary.txt"
        if s.exists():
            z.write(s, arcname="summary.txt")
    return str(zp)

