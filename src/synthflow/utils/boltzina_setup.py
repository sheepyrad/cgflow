"""Setup utility for Boltzina integration with CGFlow.

This module handles pre-training setup including:
1. Predicting protein structure using Boltz-2
2. Docking reference ligand to calculate grid box
3. Combining structures for Boltzina
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from Bio.PDB import PDBIO, PDBParser

from synthflow.utils import unidock
from synthflow.utils.conda_env import run_in_conda_env
from synthflow.utils.extract_pocket import get_mol_center


def generate_grid_config(
    pdb_path: str | Path,
    residues: list[str],
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Generate grid configuration based on PDB file and targeted residues.

    Parameters
    ----------
    pdb_path : str | Path
        Path to PDB file
    residues : list[str]
        List of residues for targeted docking (format: ['A:123', 'B:456'])
    output_dir : Optional[Path]
        Directory to save config file

    Returns
    -------
    dict with config_path, grid_dimensions, config_text

    Raises
    ------
    ValueError
        If no residues specified or no atoms found
    RuntimeError
        If grid generation fails
    """
    if not residues:
        raise ValueError("target_residues must be provided for targeted grid box generation.")

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)

        coords = []
        # Extract coordinates of specific residues
        for residue_spec in residues:
            residue_spec = residue_spec.strip()
            chain_id, res_id = residue_spec.split(":")
            chain_id = chain_id.strip()
            res_id = res_id.strip()

            for chain in structure.get_chains():
                if chain.id == chain_id:
                    for res in chain.get_residues():
                        if res.id[1] == int(res_id):  # Match residue number
                            for atom in res:
                                coords.append(atom.coord)

        if not coords:
            raise ValueError("No atoms found for the specified residues.")

        coords = np.array(coords)
        min_coords = coords.min(axis=0) - 5  # Add buffer
        max_coords = coords.max(axis=0) + 5  # Add buffer

        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords

        # Create configuration file for grid box
        config_text = f"""center_x = {center[0]}
center_y = {center[1]}
center_z = {center[2]}
size_x = {size[0]}
size_y = {size[1]}
size_z = {size[2]}"""

        # Generate a unique filename using timestamp
        timestamp = int(time.time())
        config_filename = f"config_targeted_{timestamp}.txt"

        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        config_path = output_dir / config_filename

        with open(config_path, "w") as f:
            f.write(config_text)

        # Extract grid dimensions to return
        grid_dimensions = {
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "center_z": float(center[2]),
            "size_x": float(size[0]),
            "size_y": float(size[1]),
            "size_z": float(size[2]),
        }

        return {
            "config_path": str(config_path),
            "config_filename": config_filename,
            "grid_dimensions": grid_dimensions,
            "config_text": config_text,
        }

    except Exception as e:
        raise RuntimeError(f"Error during grid generation: {str(e)}") from e


def predict_protein_structure(
    base_yaml: str | Path,
    output_dir: Path,
    use_msa_server: bool = False,
) -> Path:
    """
    Predict protein structure using Boltz-2.

    Parameters
    ----------
    base_yaml : str | Path
        Path to Boltz-2 base.yaml config file
    output_dir : Path
        Output directory for Boltz-2 predictions
    use_msa_server : bool
        Whether to use MSA server

    Returns
    -------
    Path to predicted receptor PDB file
    """
    base_yaml = Path(base_yaml)
    # Resolve to absolute path if relative
    if not base_yaml.is_absolute():
        base_yaml = base_yaml.resolve()
    
    if not base_yaml.exists():
        raise FileNotFoundError(f"Boltz-2 base.yaml file not found: {base_yaml}")
    
    output_dir = Path(output_dir)
    # Resolve to absolute path to avoid nested directory issues
    if not output_dir.is_absolute():
        output_dir = output_dir.resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a temporary modified base_yaml with a dummy ligand chain B
    # This ensures the manifest.json includes chain B and affinity info from the start
    # The decoy ligand won't interfere with scoring since we use our own docked ligands
    with open(base_yaml, "r") as f:
        yaml_data = yaml.safe_load(f)
    
    # Check if ligand chain B already exists
    has_ligand_chain = False
    if "sequences" in yaml_data:
        for seq in yaml_data["sequences"]:
            if "ligand" in seq:
                ligand_id = seq["ligand"].get("id") if isinstance(seq["ligand"], dict) else None
                if ligand_id == "B":
                    has_ligand_chain = True
                    break
    
    # Add dummy ligand chain B if it doesn't exist
    if not has_ligand_chain:
        if "sequences" not in yaml_data:
            yaml_data["sequences"] = []
        
        # Add dummy ligand (single carbon atom) - minimal, won't interfere
        yaml_data["sequences"].append({
            "ligand": {
                "id": "B",
                "smiles": "C"  # Dummy: single carbon atom
            }
        })
        
        # Add affinity property if not present
        if "properties" not in yaml_data:
            yaml_data["properties"] = []
        
        has_affinity = any(
            prop.get("affinity", {}).get("binder") == "B"
            for prop in yaml_data.get("properties", [])
        )
        
        if not has_affinity:
            yaml_data["properties"].append({
                "affinity": {
                    "binder": "B"
                }
            })
        
        # Write modified YAML to temporary file
        temp_yaml = output_dir / "base_yaml_with_decoy_ligand.yaml"
        with open(temp_yaml, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        
        print(f"Created temporary base_yaml with decoy ligand chain B: {temp_yaml}")
        base_yaml = temp_yaml  # Use modified YAML for prediction

    # Run Boltz-2 prediction in boltzina conda environment
    # Use absolute paths to avoid nested directory creation
    # Note: MSA should be specified in the base.yaml file, so we don't use --use_msa_server
    cmd = ["boltz", "predict", str(base_yaml), "--out_dir", str(output_dir), "--output_format", "pdb"]
    # Only add --use_msa_server if explicitly requested (usually not needed if MSA is in yaml)
    if use_msa_server:
        cmd.append("--use_msa_server")

    try:
        # Use absolute paths and run from a stable directory to avoid path duplication
        # Don't change cwd - let Boltz-2 handle paths as absolute
        result = run_in_conda_env(
            cmd,
            conda_env="boltzina",
            cwd=None,  # Use current working directory (where script is run from)
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Boltz-2 prediction completed: {result.stdout}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Boltz-2 prediction failed: {e.stderr}") from e

    # Find the predicted receptor PDB file
    # Boltz-2 creates output in <out_dir>/predictions/ or might create nested structure
    # First check the expected location
    predictions_dir = output_dir / "predictions"
    if not predictions_dir.exists():
        # Boltz-2 might have created a nested structure - search recursively
        all_predictions = list(output_dir.rglob("predictions"))
        if all_predictions:
            # Use the first predictions directory found
            predictions_dir = all_predictions[0]
        else:
            # Check if predictions are directly in output_dir
            pdb_files = list(output_dir.rglob("*_protein.pdb"))
            if pdb_files:
                # If PDB files exist directly, use output_dir as predictions_dir
                predictions_dir = output_dir
            else:
                raise RuntimeError(
                    f"Boltz-2 predictions directory not found in: {output_dir}\n"
                    f"Output directory contents: {list(output_dir.iterdir()) if output_dir.exists() else 'not found'}\n"
                    f"This usually means Boltz-2 prediction failed. Check the error messages above."
                )

    # Look for protein PDB files
    # Boltz-2 outputs PDB files in predictions/<target_name>/<target_name>_model_*.pdb
    # or predictions/<target_name>/<target_name>_model_*_protein.pdb format
    receptor_files = list(predictions_dir.rglob("*_model_*_protein.pdb"))
    if not receptor_files:
        # Try alternative naming: <target_name>_model_0.pdb (without _protein suffix)
        receptor_files = list(predictions_dir.rglob("*_model_*.pdb"))
    if not receptor_files:
        # Fallback: look for any .pdb files in predictions directory
        receptor_files = list(predictions_dir.rglob("*.pdb"))
        # Filter out ligand files if any (they usually have different naming)
        receptor_files = [f for f in receptor_files if "_ligand" not in f.name.lower()]

    if not receptor_files:
        raise RuntimeError(
            f"No receptor PDB file found in {predictions_dir}\n"
            f"Contents of predictions directory: {list(predictions_dir.iterdir()) if predictions_dir.exists() else 'not found'}"
        )

    # Use the first found receptor file (usually model_0)
    receptor_pdb = receptor_files[0]
    return receptor_pdb


def dock_reference_ligand(
    receptor_pdb: Path,
    ref_ligand_path: Path,
    grid_center: tuple[float, float, float],
    grid_size: tuple[float, float, float],
    output_dir: Path,
    seed: int = 1,
) -> tuple[Path, float]:
    """
    Dock reference ligand to predicted structure using UniDock.

    Parameters
    ----------
    receptor_pdb : Path
        Path to receptor PDB file
    ref_ligand_path : Path
        Path to reference ligand file
    grid_center : tuple[float, float, float]
        Center coordinates for docking box
    grid_size : tuple[float, float, float]
        Size of docking box
    output_dir : Path
        Output directory for docked structure
    seed : int
        Random seed for docking

    Returns
    -------
    tuple of (docked_ligand_path, docking_score)
    """
    from rdkit import Chem

    # Load reference ligand
    ref_ligand_path = Path(ref_ligand_path)
    if ref_ligand_path.suffix == ".sdf":
        ref_mol = next(Chem.SDMolSupplier(str(ref_ligand_path), sanitize=False))
    elif ref_ligand_path.suffix == ".mol2":
        ref_mol = Chem.MolFromMol2File(str(ref_ligand_path), sanitize=False)
    elif ref_ligand_path.suffix == ".pdb":
        ref_mol = Chem.MolFromPDBFile(str(ref_ligand_path), sanitize=False)
    else:
        raise ValueError(f"Unsupported ligand format: {ref_ligand_path.suffix}")

    if ref_mol is None:
        raise RuntimeError(f"Failed to load reference ligand: {ref_ligand_path}")

    # Dock using UniDock
    try:
        print(f"Docking parameters:")
        print(f"  Receptor: {receptor_pdb}")
        print(f"  Ligand: {ref_ligand_path}")
        print(f"  Grid center: {grid_center}")
        print(f"  Grid size: {grid_size}")
        print(f"  Seed: {seed}")
        
        res = unidock.docking(
            [ref_mol],
            receptor_pdb,
            grid_center,
            seed=seed,
            size=grid_size,
            search_mode="balance",
            debug_dir=output_dir,  # Save config for debugging
        )
        docked_mol, docking_score = res[0]

        if docked_mol is None:
            raise RuntimeError("Docking failed: no valid pose returned")

        print(f"Docking succeeded! Score: {docking_score}")

        # Save docked ligand as SDF first (for verification)
        output_dir.mkdir(parents=True, exist_ok=True)
        docked_ligand_sdf = output_dir / "docked_ref_ligand.sdf"
        with Chem.SDWriter(str(docked_ligand_sdf)) as w:
            w.write(docked_mol)
        print(f"Saved docked ligand SDF for verification: {docked_ligand_sdf}")

        # Convert SDF to MOL2 using OpenBabel (for CGFlow)
        try:
            from openbabel import pybel
            
            # Read SDF using OpenBabel
            mol_block = Chem.MolToMolBlock(docked_mol)
            pbmol = pybel.readstring("sdf", mol_block)
            
            # Write as MOL2
            docked_ligand_mol2 = output_dir / "docked_ref_ligand.mol2"
            pbmol.write("mol2", str(docked_ligand_mol2), overwrite=True)
            print(f"Saved docked ligand MOL2 for CGFlow: {docked_ligand_mol2}")
            
            return docked_ligand_mol2, docking_score
        except ImportError:
            raise RuntimeError("OpenBabel is required to convert docked ligand to MOL2 format. Please install openbabel-wheel.")
        except Exception as e:
            raise RuntimeError(f"Failed to convert docked ligand to MOL2: {str(e)}") from e

    except Exception as e:
        print(f"Docking error details: {type(e).__name__}: {str(e)}")
        raise RuntimeError(f"Docking failed: {str(e)}") from e


def setup_boltzina_environment(
    base_yaml: str | Path,
    ref_ligand_path: str | Path | None,
    result_dir: str | Path,
    target_residues: list[str],
    protein_pdb: Optional[str | Path] = None,
    work_dir: Optional[str | Path] = None,
    use_msa_server: bool = True,
    seed: int = 1,
) -> dict:
    """
    Setup Boltzina environment for CGFlow training.

    This function:
    1. Predicts protein structure using Boltz-2 (from base.yaml with protein sequence)
    2. Generates grid box from targeted residues on the predicted structure
    3. Saves all intermediate files
    
    Grid center and size are calculated from target residues and passed directly to CGFlow.
    No reference ligand docking is performed.

    Parameters
    ----------
    base_yaml : str | Path
        Path to Boltz-2 base.yaml config file. This file contains:
        - Protein sequence (not a PDB file path)
        - Ligand SMILES (optional)
        - Properties to predict (e.g., affinity)
        Example format:
        ```yaml
        version: 1
        sequences:
        - protein:
            id: [A]
            sequence: MENFQKVEKIGEGTYGVVYK...
        ```
    ref_ligand_path : str | Path | None
        (Optional) Path to reference ligand. Not used for docking - grid center
        and size are calculated from target residues instead.
    result_dir : str | Path
        Directory to save all intermediate files
    target_residues : list[str]
        List of residues for targeted grid box generation (format: ['A:123', 'B:456'])
        Required for targeted docking. Residue numbers should match the predicted
        structure from Boltz-2.
    protein_pdb : Optional[str | Path]
        (DEPRECATED/UNUSED) Path to input protein PDB file. This parameter is not
        currently used - the protein structure is predicted from base.yaml.
        Kept for backward compatibility but can be omitted.
    work_dir : Optional[str | Path]
        Working directory for Boltz-2 prediction output (auto-generated if not provided).
        This directory must contain:
        - processed/manifest.json (required for Boltzina scoring)
        - processed/constraints/ (recommended for Boltzina scoring)
        - predictions/ (contains predicted receptor structure)
        If not provided, a new directory will be created inside boltzina_setup.
    use_msa_server : bool
        Whether to use MSA server for Boltz-2 (default: False, MSA should be specified in base.yaml)
    seed : int
        Random seed for docking

    Returns
    -------
    dict with keys:
        - receptor_pdb: Path to predicted receptor PDB (from Boltz-2)
        - grid_center: tuple[float, float, float]
        - grid_size: tuple[float, float, float]
        - work_dir: Path to Boltz-2 work directory
        - docked_ref_ligand: Path to docked reference ligand (if docked)
        - grid_config: Path to grid config file

    Raises
    ------
    ValueError
        If target_residues is not provided
    """
    if not target_residues:
        raise ValueError("target_residues must be provided for targeted grid box generation.")

    result_dir = Path(result_dir)
    # Don't create result_dir here - let the trainer create it
    # We only create subdirectories
    setup_dir = result_dir / "boltzina_setup"
    # Create parent directory if it doesn't exist (for the setup subdirectory)
    setup_dir.parent.mkdir(parents=True, exist_ok=True)
    setup_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Predict protein structure using Boltz-2
    # work_dir must be inside setup_dir to persist after trainer initialization
    # If work_dir is provided as absolute path outside setup_dir, create a copy inside setup_dir
    if work_dir is None:
        work_dir = setup_dir / "boltz_work"
    else:
        work_dir = Path(work_dir)
        # If work_dir is absolute and outside setup_dir, create a copy inside setup_dir
        # This ensures it persists after trainer overwrites log_dir
        if work_dir.is_absolute() and not str(work_dir).startswith(str(setup_dir)):
            # Check if work_dir exists and has Boltz-2 files
            if work_dir.exists() and (work_dir / "processed" / "manifest.json").exists():
                # Copy the entire work_dir to setup_dir/boltz_work
                import shutil
                new_work_dir = setup_dir / "boltz_work"
                if new_work_dir.exists():
                    shutil.rmtree(new_work_dir)
                shutil.copytree(work_dir, new_work_dir)
                work_dir = new_work_dir
                print(f"Copied Boltz-2 work directory to: {work_dir}")
            else:
                # Use a new work_dir inside setup_dir
                work_dir = setup_dir / "boltz_work"
        else:
            # Relative path or inside setup_dir - resolve relative to setup_dir
            if not work_dir.is_absolute():
                work_dir = setup_dir / work_dir
    
    # Resolve to absolute path to avoid nested directory issues
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if Boltz-2 predictions already exist
    # First check direct structure, then nested structure
    # Look for both manifest.json and constraints directory
    manifest_path = work_dir / "processed" / "manifest.json"
    constraints_dir = work_dir / "processed" / "constraints"
    
    if not manifest_path.exists():
        # Try nested structure
        manifest_files = list(work_dir.rglob("processed/manifest.json"))
        if manifest_files:
            manifest_path = manifest_files[0]
            # Use the directory containing processed/ as the actual work_dir
            actual_work_dir = manifest_path.parent.parent
            constraints_dir = actual_work_dir / "processed" / "constraints"
            print(f"Found existing Boltz-2 predictions in nested directory: {actual_work_dir}")
            # Find receptor PDB from existing predictions
            predictions_dir = actual_work_dir / "predictions"
        else:
            actual_work_dir = None
            predictions_dir = None
            constraints_dir = None
    else:
        actual_work_dir = work_dir
        predictions_dir = work_dir / "predictions"
    
    # Verify both manifest.json and constraints exist for scoring
    if manifest_path.exists() and constraints_dir and constraints_dir.exists() and predictions_dir and predictions_dir.exists():
        print(f"Using existing Boltz-2 predictions in: {actual_work_dir}")
        print(f"  - manifest.json: {manifest_path}")
        print(f"  - constraints directory: {constraints_dir}")
        receptor_files = list(predictions_dir.rglob("*_model_*.pdb"))
        if not receptor_files:
            receptor_files = list(predictions_dir.rglob("*.pdb"))
        if receptor_files:
            receptor_pdb = receptor_files[0]
            print(f"Found existing receptor: {receptor_pdb}")
            # Update work_dir to actual_work_dir for later use
            work_dir = actual_work_dir
        else:
            # Need to run prediction
            receptor_pdb = None
    else:
        receptor_pdb = None
    
    # Run prediction if needed
    if receptor_pdb is None:
        print("Predicting protein structure with Boltz-2...")
        # MSA is specified in base.yaml, so set use_msa_server=False by default
        receptor_pdb = predict_protein_structure(base_yaml, work_dir, use_msa_server=False)
        print(f"Receptor structure saved to: {receptor_pdb}")

    # Find the actual Boltz-2 output directory (may be nested)
    # Boltz-2 creates a subdirectory (e.g., boltz_results_<name>) inside work_dir
    # The actual work_dir for scoring is the one containing processed/manifest.json and processed/constraints
    actual_work_dir = None
    manifest_path = work_dir / "processed" / "manifest.json"
    constraints_dir = work_dir / "processed" / "constraints"
    
    if manifest_path.exists() and constraints_dir.exists():
        # Direct structure - work_dir is the actual output directory
        actual_work_dir = work_dir
    else:
        # Nested structure - search for processed/manifest.json
        manifest_files = list(work_dir.rglob("processed/manifest.json"))
        if manifest_files:
            # Use the parent of processed/ directory as the actual work_dir
            actual_work_dir = manifest_files[0].parent.parent
            print(f"Found Boltz-2 output directory (nested): {actual_work_dir}")
            manifest_path = actual_work_dir / "processed" / "manifest.json"
            constraints_dir = actual_work_dir / "processed" / "constraints"
        else:
            raise RuntimeError(
                f"Boltz-2 work directory is missing required files:\n"
                f"  Expected: {manifest_path} or nested structure\n"
                f"  This file is required for Boltzina scoring.\n"
                f"  Searched in: {work_dir}\n"
                f"  Please ensure Boltz-2 prediction completed successfully."
            )
    
    # Verify manifest.json exists
    if not manifest_path.exists():
        raise RuntimeError(
            f"Boltz-2 manifest.json not found at: {manifest_path}\n"
            f"  Work directory: {actual_work_dir}\n"
            f"  This file is required for Boltzina scoring."
        )
    
    # Update work_dir to the actual Boltz-2 output directory
    work_dir = actual_work_dir.resolve()
    manifest_path = work_dir / "processed" / "manifest.json"
    constraints_dir = work_dir / "processed" / "constraints"
    
    print(f"Using Boltz-2 output directory for scoring: {work_dir}")
    print(f"  - manifest.json: {manifest_path}")
    if constraints_dir.exists():
        print(f"  - constraints directory: {constraints_dir}")
        # List constraint files for verification
        constraint_files = list(constraints_dir.glob("*.npz"))
        if constraint_files:
            print(f"    Found {len(constraint_files)} constraint file(s)")
    else:
        print(f"  - Warning: constraints directory not found at {constraints_dir} (may still work)")

    # Copy receptor to setup directory for easy access
    receptor_copy = setup_dir / "predicted_receptor.pdb"
    import shutil

    shutil.copy2(receptor_pdb, receptor_copy)

    # Step 2: Generate grid box from targeted residues
    print("Calculating grid box from targeted residues...")
    grid_config = generate_grid_config(
        receptor_pdb, target_residues, setup_dir
    )
    grid_center = (
        grid_config["grid_dimensions"]["center_x"],
        grid_config["grid_dimensions"]["center_y"],
        grid_config["grid_dimensions"]["center_z"],
    )
    grid_size = (
        grid_config["grid_dimensions"]["size_x"],
        grid_config["grid_dimensions"]["size_y"],
        grid_config["grid_dimensions"]["size_z"],
    )

    print(f"Grid center: {grid_center}")
    print(f"Grid size: {grid_size}")

    # Step 4: Save setup info
    # No need to dock reference ligand - CGFlow can use grid center and size directly
    # Ensure all paths are absolute to avoid issues when trainer deletes/recreates directories
    # work_dir is the Boltz-2 prediction directory required for Boltzina scoring
    setup_info = {
        "receptor_pdb": str(receptor_pdb.resolve()),
        "receptor_copy": str(receptor_copy.resolve()),
        "grid_center": grid_center,
        "grid_size": grid_size,
        "work_dir": str(work_dir.resolve()),  # Boltz-2 prediction directory with manifest.json and constraints
        "grid_config": str(Path(grid_config["config_path"]).resolve()),
    }

    setup_info_path = setup_dir / "setup_info.json"
    with open(setup_info_path, "w") as f:
        json.dump(setup_info, f, indent=2)

    print(f"Setup complete! Info saved to: {setup_info_path}")
    print(f"Boltz-2 work directory (for scoring): {work_dir}")

    return setup_info

