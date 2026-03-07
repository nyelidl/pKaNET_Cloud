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


# =========================================
# Open Babel availability check
# =========================================
_OBABEL_AVAILABLE = None

def check_obabel():
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
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0 and Path(mol2_path).exists()
    except subprocess.TimeoutExpired:
        print("Open Babel conversion timed out")
        return False
    except Exception as e:
        print(f"Open Babel conversion error: {e}")
        return False


# =========================================
# Charge profile helper
# =========================================
def charge_profile_from_smiles(smiles: str) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES for charge profiling.")
    atom_charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    n_pos = sum(c > 0 for c in atom_charges)
    n_neg = sum(c < 0 for c in atom_charges)
    net = int(sum(atom_charges))
    return {
        "net_charge":          net,
        "has_pos":             n_pos > 0,
        "has_neg":             n_neg > 0,
        "n_pos_atoms":         int(n_pos),
        "n_neg_atoms":         int(n_neg),
        "is_zwitterion_strict": bool((n_pos > 0) and (n_neg > 0) and (net == 0)),
    }


# =========================================
# IUPAC pKa dataset
# =========================================
_IUPAC_DF     = None
_PKA_MAP_ALL  = None
_IUPAC_LOADED = False

def load_iupac_dataset():
    global _IUPAC_DF, _PKA_MAP_ALL, _IUPAC_LOADED
    if _IUPAC_LOADED:
        return
    _IUPAC_LOADED = True
    try:
        import pandas as pd
        IUPAC_CSV_URL = (
            "https://raw.githubusercontent.com/IUPAC/Dissociation-Constants"
            "/main/iupac_high-confidence_v2_3.csv"
        )
        print("⏳ Loading IUPAC pKa dataset...")
        iupac_df = pd.read_csv(IUPAC_CSV_URL)
        print(f"✓ IUPAC dataset loaded: {len(iupac_df):,} rows")

        cols      = list(iupac_df.columns)
        lower_map = {c.lower(): c for c in cols}

        smiles_col = None
        for w in ["SMILES", "smiles"]:
            if w in cols:            smiles_col = w; break
            if w.lower() in lower_map: smiles_col = lower_map[w.lower()]; break

        pka_col = None
        for w in ["pka_value", "pKa", "pka", "value"]:
            if w in cols:            pka_col = w; break
            if w.lower() in lower_map: pka_col = lower_map[w.lower()]; break

        if smiles_col is None or pka_col is None:
            print("⚠️ Cannot find SMILES/pKa columns in IUPAC dataset")
            return

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

        _IUPAC_DF    = iupac_df
        _PKA_MAP_ALL = pka_map
        print(f"✓ IUPAC indexed molecules: {len(pka_map):,}")

    except Exception as e:
        print(f"⚠️ IUPAC dataset load failed: {e}")
        print("   Will use pKaPredict only.")


def lookup_pka_iupac_stats(query_smiles: str) -> Optional[Dict[str, Any]]:
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
        "pka_median":       float(np.median(vals)),
        "pka_mean":         float(np.mean(vals)),
        "pka_min":          float(np.min(vals)),
        "pka_max":          float(np.max(vals)),
        "n":                len(vals),
        "all":              vals,
        "canonical_smiles": canonical_smi,
    }


# =========================================
# pKaPredict ML model
# =========================================
_PKANET_MODEL      = None
_DESCRIPTOR_NAMES  = None

def get_model():
    global _PKANET_MODEL, _DESCRIPTOR_NAMES
    if _PKANET_MODEL is None:
        _PKANET_MODEL = load_model()
        if hasattr(_PKANET_MODEL, 'feature_name_'):
            _DESCRIPTOR_NAMES = _PKANET_MODEL.feature_name_
        else:
            from rdkit.Chem import Descriptors
            all_descriptors   = [desc[0] for desc in Descriptors._descList]
            _DESCRIPTOR_NAMES = all_descriptors[:_PKANET_MODEL.n_features_]
        print(f"✓ pKaPredict model loaded ({len(_DESCRIPTOR_NAMES)} descriptors)")
    return _PKANET_MODEL, _DESCRIPTOR_NAMES


def predict_pka_pkanet(smiles: str) -> float:
    smiles = smiles.strip()
    mol    = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES for pKa prediction.")
    model, descriptor_names = get_model()
    pka_value = predict_pKa(smiles, model, descriptor_names)
    if isinstance(pka_value, (list, tuple)):
        pka_value = pka_value[0]
    elif hasattr(pka_value, '__iter__') and not isinstance(pka_value, str):
        pka_value = next(iter(pka_value))
    return float(pka_value)


# =========================================
# pH adjustment via Dimorphite-DL
# =========================================
def ph_adjust_smiles_dimorphite(
    smiles_str: str, ph: float, mode: str = "AUTO"
) -> Tuple[str, int, Dict[str, Any]]:
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
        for smi, prof in candidates:
            if prof["is_zwitterion_strict"]:
                return smi, prof["net_charge"], prof
        smi, prof = min(candidates, key=lambda x: abs(x[1]["net_charge"]))
        return smi, prof["net_charge"], prof

    if mode == "NORMAL":
        smi, prof = min(candidates, key=lambda x: abs(x[1]["net_charge"]))
        return smi, prof["net_charge"], prof

    # AUTO — first variant
    smi, prof = candidates[0]
    return smi, prof["net_charge"], prof


# =========================================
# 3D structure generation
# =========================================
def build_minimized_3d(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES for 3D build.")
    mol  = Chem.AddHs(mol)
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


# =========================================
# Misc helpers
# =========================================
def parse_smi_lines(text: str):
    records = []
    idx = 1
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        smi   = parts[0]
        name  = parts[1] if len(parts) > 1 else f"mol_{idx:03d}"
        records.append((smi, name))
        idx += 1
    return records


def generate_RS_variants(base_smiles: str, base_name: str, keep_original: bool = False):
    mol = Chem.MolFromSmiles(base_smiles)
    if mol is None:
        return [{"name": base_name, "stereo": None, "base_smiles": base_smiles}]

    if keep_original:
        return [{"name": base_name, "stereo": None,
                 "base_smiles": Chem.MolToSmiles(mol, isomericSmiles=True)}]

    opts    = StereoEnumerationOptions(onlyUnassigned=False)
    isomers = list(EnumerateStereoisomers(mol, options=opts))

    if len(isomers) == 1:
        return [{"name": base_name, "stereo": None,
                 "base_smiles": Chem.MolToSmiles(isomers[0], isomericSmiles=True)}]

    variants = []
    used     = set()
    for iso in isomers:
        Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
        centers   = Chem.FindMolChiralCenters(iso, includeUnassigned=False)
        labs      = {lab for _, lab in centers if lab in ("R", "S")}
        label_here = None
        if "R" in labs and "R" not in used:   label_here = "R"
        elif "S" in labs and "S" not in used: label_here = "S"
        if label_here:
            used.add(label_here)
            variants.append({
                "name":        base_name,
                "stereo":      label_here,
                "base_smiles": Chem.MolToSmiles(iso, isomericSmiles=True),
            })
        if used == {"R", "S"}:
            break

    return variants if variants else [
        {"name": base_name, "stereo": None,
         "base_smiles": Chem.MolToSmiles(isomers[0], isomericSmiles=True)}
    ]


def save_2d_structure_image(smiles: str, output_path: str, size=(800, 600)) -> bool:
    try:
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        AllChem.Compute2DCoords(mol)
        Draw.MolToImage(mol, size=size).save(output_path)
        return True
    except Exception as e:
        print(f"Warning: 2D structure image generation failed: {e}")
        return False


def save_molecule_files(mol, base_path: str, formats: List[str]) -> Dict[str, Any]:
    saved_files = {}
    warnings    = []
    mol2_via_obabel = False

    # SDF always generated
    try:
        sdf_path = f"{base_path}.sdf"
        writer   = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        saved_files["sdf"] = sdf_path
    except Exception as e:
        warnings.append(f"Could not save SDF format: {e}")

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
                if hasattr(Chem, 'MolToMol2File'):
                    try:
                        Chem.MolToMol2File(mol, file_path)
                        saved_files["mol2"] = file_path
                        continue
                    except Exception:
                        pass

                # Fallback: PDB → MOL2 via Open Babel
                if "pdb" not in saved_files:
                    pdb_path = f"{base_path}.pdb"
                    try:
                        Chem.MolToPDBFile(mol, pdb_path)
                        saved_files["pdb"] = pdb_path
                    except Exception as e:
                        warnings.append(f"Could not generate PDB for MOL2 conversion: {e}")
                        continue

                pdb_path = saved_files.get("pdb")
                if pdb_path and convert_pdb_to_mol2_obabel(pdb_path, file_path):
                    saved_files["mol2"] = file_path
                    mol2_via_obabel     = True
                else:
                    if not check_obabel():
                        warnings.append(
                            "MOL2 format not available. "
                            "Install Open Babel (obabel) to enable MOL2 output."
                        )
                    else:
                        warnings.append("MOL2 conversion failed. Using PDB format instead.")

        except Exception as e:
            warnings.append(f"Could not save {fmt_upper} format: {e}")

    if mol2_via_obabel:
        warnings.append("ℹ️ MOL2 files generated using Open Babel (converted from PDB)")

    return {"files": saved_files, "warnings": warnings}


# =========================================
# Main job runner
# =========================================
def run_job(
    *,
    input_type:              str,
    smiles_text:             str | None,
    uploaded_bytes:          bytes | None,
    uploaded_name:           str | None,
    target_pH:               float,
    output_name:             str,
    out_dir:                 str,
    output_formats:          List[str] = None,
    enumerate_stereoisomers: bool = True,
    charge_mode:             str = "AUTO",
    use_iupac_pka:           bool = True,
    use_xtb_pka:             bool = False,
    xtb_results_map:         dict | None = None,
) -> Dict[str, Any]:

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not output_formats:
        output_formats = ["PDB"]
    formats_to_save = [fmt.upper() for fmt in output_formats]

    # Load IUPAC dataset if requested
    if use_iupac_pka and not _IUPAC_LOADED:
        try:
            load_iupac_dataset()
        except Exception:
            pass

    # ── Parse inputs ──────────────────────────────────────────────────────
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
        ext      = os.path.splitext(uploaded_name)[1].lower()
        tmp_path = out / f"uploaded{ext}"
        tmp_path.write_bytes(uploaded_bytes)

        mol_in = None
        if ext == ".pdb":
            mol_in = Chem.MolFromPDBFile(str(tmp_path), removeHs=False, sanitize=False)
        elif ext == ".mol2":
            mol_in = Chem.MolFromMol2File(str(tmp_path), removeHs=False, sanitize=False)
        elif ext == ".sdf":
            supplier = Chem.SDMolSupplier(str(tmp_path), removeHs=False, sanitize=False)
            mol_in   = next((m for m in supplier if m is not None), None)
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
        ligands_raw.append({
            "name":        output_name or os.path.splitext(uploaded_name)[0],
            "base_smiles": base_smiles,
        })
    else:
        raise ValueError("Unknown input_type")

    # ── Enumerate stereoisomers ───────────────────────────────────────────
    ligands     = []
    keep_stereo = not enumerate_stereoisomers
    for lig in ligands_raw:
        ligands.extend(generate_RS_variants(lig["base_smiles"], lig["name"], keep_stereo))

    results         = []
    format_warnings = []

    for lig in ligands:
        base_name   = lig["name"]
        stereo      = lig.get("stereo")
        suffix      = f"_{stereo}" if stereo else ""
        pretty_name = base_name + suffix
        base_smiles = lig["base_smiles"]

        # ── pKa Prediction ────────────────────────────────────────────────
        # Decision matrix:
        #
        #  use_iupac | use_xtb | IUPAC matched? | Result
        #  ─────────────────────────────────────────────────────────────────
        #  True      | False   | Yes            | IUPAC pKa
        #  True      | False   | No             | pKaPredict ML
        #  True      | True    | Yes            | IUPAC pKa  (xTB shown alongside in app)
        #  True      | True    | No             | xTB only   (skip ML)
        #  False     | True    | —              | xTB only   (skip ML)
        #  False     | False   | —              | pKaPredict ML
        # ─────────────────────────────────────────────────────────────────
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
                    print(f"pKa (IUPAC n={stats['n']}) for {pretty_name}: {pka_pred:.2f}")

            if pka_pred is None:
                # No IUPAC match (or IUPAC disabled)
                if use_xtb_pka:
                    # Skip ML — xTB is the sole source
                    pka_source = "xTB"
                    print(f"No IUPAC match for {pretty_name} — xTB pKa will be used.")
                else:
                    pka_pred   = predict_pka_pkanet(base_smiles)
                    pka_source = "pKaPredict (ML)"
                    print(f"pKa (pKaPredict ML) for {pretty_name}: {pka_pred:.2f}")

        except Exception as e:
            print(f"Warning: pKa prediction failed for {pretty_name}: {e}")
            warning_msg = f"pKa prediction failed for {pretty_name}: {str(e)}"
            if warning_msg not in format_warnings:
                format_warnings.append(warning_msg)

        # ── pH adjustment ─────────────────────────────────────────────────
        try:
            ph_smiles, formal_charge, charge_prof = ph_adjust_smiles_dimorphite(
                base_smiles, target_pH, charge_mode
            )
        except Exception as e:
            print(f"Error: pH adjustment failed for {pretty_name}: {e}")
            continue

        # ── 3D generation ─────────────────────────────────────────────────
        mol_min = build_minimized_3d(ph_smiles)

        # ── Save files ────────────────────────────────────────────────────
        base_file_path = str(out / f"{base_name}{suffix}_min")
        save_result    = save_molecule_files(mol_min, base_file_path, formats_to_save)
        saved_files    = save_result["files"]

        png_path = str(out / f"{base_name}{suffix}_2D.png")
        if save_2d_structure_image(ph_smiles, png_path):
            saved_files["png_2d"] = png_path

        for warning in save_result["warnings"]:
            if warning not in format_warnings:
                format_warnings.append(warning)

        # ── Build result entry ────────────────────────────────────────────
        result_entry = {
            "name":               pretty_name,
            "base_smiles":        base_smiles,
            "ph_smiles":          ph_smiles,
            "pka_pred":           pka_pred,
            "pka_source":         pka_source,
            "pka_n":              pka_n,
            "pka_iupac_matched":  pka_iupac_matched,
            "formal_charge":      formal_charge,
            "has_pos":            charge_prof["has_pos"],
            "has_neg":            charge_prof["has_neg"],
            "n_pos_atoms":        charge_prof["n_pos_atoms"],
            "n_neg_atoms":        charge_prof["n_neg_atoms"],
            "is_zwitterion":      charge_prof["is_zwitterion_strict"],
        }

        if stereo:
            result_entry["stereoisomer_id"] = stereo

        if "pdb"  in saved_files: result_entry["minimized_pdb"]  = saved_files["pdb"]
        if "sdf"  in saved_files: result_entry["minimized_sdf"]  = saved_files["sdf"]
        if "mol2" in saved_files: result_entry["minimized_mol2"] = saved_files["mol2"]

        results.append(result_entry)

    # ── Summary file ──────────────────────────────────────────────────────
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("pKaNET Cloud - Analysis Summary")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Target pH                  : {target_pH}")
    summary_lines.append(f"Charge mode                : {charge_mode}")
    summary_lines.append(f"Stereoisomer enumeration   : {'Enabled' if enumerate_stereoisomers else 'Disabled'}")
    summary_lines.append(f"Total structures generated : {len(results)}")
    summary_lines.append(f"pKa strategy               : "
        + ("IUPAC → pKaPredict (ML)" if use_iupac_pka and not use_xtb_pka else
           "IUPAC → xTB (ML skipped)" if use_iupac_pka and use_xtb_pka else
           "xTB only"                 if use_xtb_pka else
           "pKaPredict (ML) only"))
    summary_lines.append("Zwitterion (strict)        : has + and − atoms AND net charge = 0")
    summary_lines.append("=" * 80)
    summary_lines.append("")

    for r in results:
        summary_lines.append(f"Molecule: {r['name']}")
        summary_lines.append("-" * 80)
        summary_lines.append(f"  Base SMILES          : {r['base_smiles']}")
        summary_lines.append(f"  pH-adjusted SMILES   : {r['ph_smiles']}")

        # pKa line
        if r["pka_pred"] is not None:
            summary_lines.append(
                f"  Predicted pKa        : {r['pka_pred']:.2f} ({r['pka_source']})"
            )
        elif r.get("pka_source") == "xTB":
            summary_lines.append(f"  Predicted pKa        : see xTB results below")
        else:
            summary_lines.append(f"  Predicted pKa        : N/A")

        # xTB pKa results (appended inline per molecule)
        if xtb_results_map:
            xtb_res = xtb_results_map.get(r["base_smiles"], [])
            for xr in xtb_res:
                if xr.get("pKa") is not None:
                    summary_lines.append(
                        f"  xTB pKa ({xr['group']:8s})  : {xr['pKa']:.1f}  "
                        f"[ΔE={xr['dE_kcal']:+.3f} kcal/mol]"
                    )
                elif xr.get("error"):
                    summary_lines.append(
                        f"  xTB pKa ({xr.get('group','?'):8s})  : ERROR — {xr['error']}"
                    )

        summary_lines.append(f"  Formal Charge (pH {target_pH}): {r['formal_charge']:+d}")
        summary_lines.append(f"  Zwitterion (strict)  : {'YES' if r.get('is_zwitterion') else 'NO'}")
        summary_lines.append(f"  + atoms / − atoms    : {r.get('n_pos_atoms', 0)} / {r.get('n_neg_atoms', 0)}")

        generated_formats = []
        if "minimized_pdb"  in r: generated_formats.append("PDB")
        if "minimized_mol2" in r: generated_formats.append("MOL2")
        if "minimized_sdf"  in r: generated_formats.append("SDF")
        summary_lines.append(f"  Output Formats       : {', '.join(generated_formats)}")

        if "stereoisomer_id" in r:
            summary_lines.append(f"  Stereoisomer         : {r['stereoisomer_id']}")
        summary_lines.append("")

    summary_lines.append("=" * 80)
    summary_text = "\n".join(summary_lines).strip()
    (out / "summary.txt").write_text(summary_text + "\n")

    # ── Processing log (SMI_FILE) ─────────────────────────────────────────
    if input_type == "SMI_FILE" and results:
        log_lines = [
            "# pKaNET Cloud - Processing Log",
            f"# Target pH: {target_pH}",
            f"# Charge mode: {charge_mode}",
            f"# Stereoisomer enumeration: {'enabled' if enumerate_stereoisomers else 'disabled'}",
            f"# Total molecules processed: {len(results)}",
            f"# pKa strategy: "
            + ("IUPAC → pKaPredict (ML)" if use_iupac_pka and not use_xtb_pka else
               "IUPAC → xTB (ML skipped)" if use_iupac_pka and use_xtb_pka else
               "xTB only"                 if use_xtb_pka else
               "pKaPredict (ML) only"),
            "#" + "=" * 70,
            "",
            "# Columns: Name | pH-adjusted SMILES | Formal Charge | "
            "Predicted pKa | pKa Source | Zwitterion | xTB pKa",
            "",
        ]
        for r in results:
            pka_str = f"{r['pka_pred']:.2f}" if r["pka_pred"] is not None else "N/A"
            pka_src = r.get("pka_source", "-")
            zw      = "Yes" if r.get("is_zwitterion") else "No"

            # Build xTB pKa string for log column
            xtb_str = "-"
            if xtb_results_map:
                xtb_res   = xtb_results_map.get(r["base_smiles"], [])
                xtb_parts = [
                    f"{xr['group']}:{xr['pKa']:.1f}"
                    for xr in xtb_res if xr.get("pKa") is not None
                ]
                if xtb_parts:
                    xtb_str = ";".join(xtb_parts)

            log_lines.append(
                f"{r['name']}\t{r['ph_smiles']}\t{r['formal_charge']:+d}\t"
                f"{pka_str}\t{pka_src}\t{zw}\t{xtb_str}"
            )

        (out / "processing.log").write_text("\n".join(log_lines) + "\n")

    return {
        "results":         results,
        "summary_text":    summary_text,
        "out_dir":         str(out),
        "format_warnings": format_warnings,
    }


# =========================================
# ZIP helpers
# =========================================
def zip_minimized_structures(out_dir: str, zip_path: str, selected_formats: List[str]) -> str:
    out            = Path(out_dir)
    zp             = Path(zip_path)
    formats_lower  = [fmt.lower() for fmt in selected_formats]
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.glob("*_min.*"):
            suffix = p.suffix.lower()
            if suffix == ".pdb"  and "pdb"  in formats_lower: z.write(p, arcname=p.name)
            elif suffix == ".mol2" and "mol2" in formats_lower: z.write(p, arcname=p.name)
    return str(zp)


def zip_all_outputs(out_dir: str, zip_path: str) -> str:
    out = Path(out_dir)
    zp  = Path(zip_path)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(out))
    return str(zp)
