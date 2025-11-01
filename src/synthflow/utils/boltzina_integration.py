"""Integration utility for Boltzina scoring in CGFlow training.

This module handles:
1. Converting docked SDF molecules to PDB format
2. Creating receptor-ligand complex files
3. Running Boltzina scoring_only mode
4. Extracting affinity predictions
"""

import json
import logging
import pickle
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

from synthflow.utils.conda_env import run_python_in_conda_env

# Ensure RDKit preserves all properties when pickling (required for Boltzina)
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


def validate_complex_cif_has_ligand(complex_cif: Path, ligand_residue_name: str = "UNL") -> bool:
    """
    Validate that a complex CIF file contains ligand atoms.
    
    Parameters
    ----------
    complex_cif : Path
        Path to complex CIF file
    ligand_residue_name : str
        Expected ligand residue name (default: "UNL")
        
    Returns
    -------
    bool
        True if ligand atoms are found, False otherwise
    """
    try:
        if not complex_cif.exists():
            return False
        
        # Read CIF file and check for ligand atoms
        with open(complex_cif, 'r') as f:
            content = f.read()
        
        # Check for HETATM lines with the ligand residue name
        # Also check for atom_site entries in CIF format
        ligand_atom_patterns = [
            f'HETATM.*{ligand_residue_name}',  # PDB format
            f'atom_site.*{ligand_residue_name}',  # CIF format
        ]
        
        # Also check for the ligand in entity_nonpoly section
        has_ligand_entity = f'_pdbx_entity_nonpoly.comp_id.*{ligand_residue_name}' in content
        
        # Count ligand atoms
        ligand_atom_count = 0
        lines = content.split('\n')
        
        # Look for atom_site loop header
        atom_site_loop_started = False
        atom_site_columns = {}
        column_index = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check for HETATM lines (PDB format in CIF)
            if 'HETATM' in line and ligand_residue_name in line:
                ligand_atom_count += 1
            
            # Check for atom_site loop
            if stripped.startswith('loop_'):
                atom_site_loop_started = True
                atom_site_columns = {}
                column_index = 0
                continue
            
            if atom_site_loop_started:
                if stripped.startswith('_atom_site.'):
                    # Extract column name
                    col_name = stripped.split()[0].replace('_atom_site.', '')
                    atom_site_columns[col_name] = column_index
                    column_index += 1
                elif stripped.startswith('_') or stripped == '':
                    # End of loop or start of new section
                    atom_site_loop_started = False
                    atom_site_columns = {}
                elif not stripped.startswith('#') and stripped:
                    # Data line - check if comp_id column contains ligand residue name
                    fields = stripped.split()
                    if 'label_comp_id' in atom_site_columns:
                        comp_id_idx = atom_site_columns['label_comp_id']
                        if comp_id_idx < len(fields) and fields[comp_id_idx] == ligand_residue_name:
                            ligand_atom_count += 1
                    elif 'auth_comp_id' in atom_site_columns:
                        comp_id_idx = atom_site_columns['auth_comp_id']
                        if comp_id_idx < len(fields) and fields[comp_id_idx] == ligand_residue_name:
                            ligand_atom_count += 1
        
        return ligand_atom_count > 0 or has_ligand_entity
        
    except Exception as e:
        print(f"Error validating complex CIF: {e}")
        return False


def setup_debug_logger(output_dir: Path) -> logging.Logger:
    """
    Set up a debug logger that writes to both file and console.
    
    Parameters
    ----------
    output_dir : Path
        Directory where debug log will be saved
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    log_file = output_dir / "boltzina_debug.log"
    
    # Create logger
    logger = logging.getLogger(f"boltzina_debug_{output_dir.name}")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler (all debug info)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def sdf_to_pdb_with_naming(mol: Chem.Mol, output_path: Path, mol_name: str = "MOL") -> bool:
    """
    Convert RDKit molecule to PDB format with proper atom naming for Boltz-2.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object
    output_path : Path
        Output PDB file path
    mol_name : str
        Residue name for the ligand (default: "MOL")

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        if mol.GetNumAtoms() == 0:
            return False

        # Ensure molecule has conformer
        if mol.GetNumConformers() == 0:
            return False

        # Assign canonical atom ordering for consistent naming
        canonical_order = AllChem.CanonicalRankAtoms(mol)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

        pdb_resn, pdb_chain, pdb_resi = mol_name, "A", 1

        for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
            # Generate atom names using element symbol + canonical index
            atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
            if len(atom_name) > 4:
                atom_name = atom_name[:4]  # Truncate if too long

            atom.SetProp("name", atom_name)
            info = atom.GetPDBResidueInfo()
            if info is None:
                info = Chem.AtomPDBResidueInfo()

            # Set PDB atom information
            info.SetName(atom_name.rjust(4))
            info.SetResidueName(pdb_resn)
            info.SetResidueNumber(pdb_resi)
            info.SetChainId(pdb_chain)
            info.SetIsHeteroAtom(True)

            atom.SetMonomerInfo(info)

        # Write PDB file
        Chem.MolToPDBFile(mol, str(output_path))
        
        # Ensure all atoms have the "name" property set (required by Boltzina's parse_ccd_residue)
        # This is critical because Boltzina expects atom.GetProp("name") to work
        for atom in mol.GetAtoms():
            if not atom.HasProp("name"):
                # Fallback: use PDB residue info if available, otherwise element + index
                info = atom.GetPDBResidueInfo()
                if info:
                    atom_name = info.GetName().strip().upper()
                    atom.SetProp("name", atom_name)
                else:
                    atom.SetProp("name", f"{atom.GetSymbol().upper()}{atom.GetIdx() + 1}")
        
        return True

    except Exception as e:
        print(f"Error converting SDF to PDB: {e}")
        return False


def generate_minimal_boltzina_config(
    work_dir: Path,
    receptor_pdb: Path,
    output_dir: Path,
    fname: str = "cgflow_ligand",
    input_ligand_name: str = "UNL",
) -> Path:
    """
    Generate minimal JSON config file for Boltzina scoring_only mode.
    
    The fname parameter should match a record ID in the manifest.json file.
    If not provided or doesn't match, it will be extracted from the manifest.

    Parameters
    ----------
    work_dir : Path
        Working directory for Boltz-2 (contains manifest.json and constraints)
    receptor_pdb : Path
        Path to receptor PDB file
    output_dir : Path
        Output directory for Boltzina results
    fname : str
        Base filename for output files (should match manifest record ID)
    input_ligand_name : str
        Name of ligand in input files

    Returns
    -------
    Path
        Path to generated config file
    """
    # Read manifest.json to get the actual record ID
    manifest_path = work_dir / "processed" / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if manifest.get("records") and len(manifest["records"]) > 0:
            # Use the first record's ID as fname
            actual_fname = manifest["records"][0]["id"]
            print(f"Using fname from manifest.json: {actual_fname} (requested: {fname})")
            fname = actual_fname
    
    config = {
        "work_dir": str(work_dir),
        "receptor_pdb": str(receptor_pdb),
        "output_dir": str(output_dir),
        "fname": fname,
        "input_ligand_name": input_ligand_name,
        "scoring_only": True,
    }

    config_path = output_dir / "boltzina_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


def _remove_chain_b_from_receptor(receptor_pdb: Path, output_path: Path) -> Path:
    """
    Remove chain B from receptor PDB file (to avoid conflicts with ligand).
    
    Parameters
    ----------
    receptor_pdb : Path
        Path to receptor PDB file
    output_path : Path
        Path to write cleaned receptor
        
    Returns
    -------
    Path
        Path to cleaned receptor file
    """
    if not receptor_pdb.exists():
        return receptor_pdb
    
    # Filter out chain B atoms from receptor PDB
    with open(receptor_pdb, "r") as f:
        receptor_lines = f.readlines()
    
    filtered_lines = []
    for line in receptor_lines:
        # Skip HETATM and ATOM lines with chain B
        if line.startswith(("ATOM", "HETATM")):
            # Chain ID is typically at position 21 (0-indexed) in PDB format
            if len(line) > 21 and line[21] == "B":
                continue
        filtered_lines.append(line)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(filtered_lines)
    
    return output_path


def create_receptor_ligand_complex(
    receptor_pdb: Path,
    ligand_pdb: Path,
    output_dir: Path,
    base_name: str,
    input_ligand_name: str = "UNL",
    base_ligand_name: str = "MOL",  # Default to "MOL" because Boltzina's parse_mmcif skips UNL ligands
    receptor_no_b: Optional[Path] = None,  # Pre-processed receptor without chain B
) -> Optional[Path]:
    """
    Create receptor-ligand complex file following Boltzina's _process_pose logic.

    Parameters
    ----------
    receptor_pdb : Path
        Path to receptor PDB file
    ligand_pdb : Path
        Path to ligand PDB file
    output_dir : Path
        Output directory for complex files
    base_name : str
        Base name for output files
    input_ligand_name : str
        Ligand name in input PDB
    base_ligand_name : str
        Base ligand name for Boltz-2
    receptor_no_b : Optional[Path]
        Pre-processed receptor without chain B (if None, will use receptor_pdb)

    Returns
    -------
    Optional[Path]
        Path to complex_fix_cif file, or None if failed
    """
    docked_ligands_dir = output_dir / "docked_ligands"
    docked_ligands_dir.mkdir(parents=True, exist_ok=True)

    prep_file = docked_ligands_dir / f"{base_name}_prep.pdb"
    complex_file = docked_ligands_dir / f"{base_name}_B_complex.pdb"
    complex_cif = docked_ligands_dir / f"{base_name}_B_complex.cif"
    complex_fix_cif = docked_ligands_dir / f"{base_name}_B_complex_fix.cif"

    try:
        # Process with pdb_chain and pdb_rplresname
        if input_ligand_name != base_ligand_name:
            cmd1 = (
                f'pdb_chain -B {ligand_pdb} | pdb_rplresname -"{input_ligand_name}":{base_ligand_name} | '
                f"pdb_tidy > {prep_file}"
            )
            subprocess.run(cmd1, shell=True, check=True)
        else:
            cmd1 = f"pdb_chain -B {ligand_pdb} | pdb_tidy > {prep_file}"
            subprocess.run(cmd1, shell=True, check=True)

        if not prep_file.exists():
            raise RuntimeError(f"Failed to create prep file: {prep_file}")

        # Use pre-processed receptor if provided, otherwise use original
        receptor_to_merge = receptor_no_b if receptor_no_b is not None and receptor_no_b.exists() else receptor_pdb

        # Merge with receptor (without chain B)
        cmd2 = f"pdb_merge {receptor_to_merge} {prep_file} | pdb_tidy > {complex_file}"
        subprocess.run(cmd2, shell=True, check=True)

        if not complex_file.exists():
            raise RuntimeError(f"Failed to create complex file: {complex_file}")

        # Convert to CIF
        cmd3 = ["maxit", "-input", str(complex_file), "-output", str(complex_cif), "-o", "1"]
        subprocess.run(cmd3, check=True)

        if not complex_cif.exists():
            raise RuntimeError(f"Failed to create CIF file: {complex_cif}")

        # Fix CIF
        cmd4 = ["maxit", "-input", str(complex_cif), "-output", str(complex_fix_cif), "-o", "8"]
        subprocess.run(cmd4, check=True)

        if not complex_fix_cif.exists():
            raise RuntimeError(f"Failed to create fixed CIF file: {complex_fix_cif}")

        # Validate that the complex CIF contains ligand atoms
        if not validate_complex_cif_has_ligand(complex_fix_cif, base_ligand_name):
            raise RuntimeError(f"Complex CIF file does not contain ligand atoms with residue name '{base_ligand_name}': {complex_fix_cif}")

        return complex_fix_cif

    except subprocess.CalledProcessError:
        return None
    except Exception:
        return None


def boltzina_scoring(
    docked_mols: list[Chem.Mol],
    receptor_pdb: Path,
    work_dir: Path,
    output_dir: Path,
    fname: str = "cgflow_ligand",
    input_ligand_name: str = "UNL",
    batch_size: int = 1,
    num_workers: int = 1,
    seed: Optional[int] = None,
    prepared_mols_dict: Optional[dict] = None,
) -> list[tuple[float, float]]:
    """
    Score docked molecules using Boltzina.

    Parameters
    ----------
    docked_mols : list[Chem.Mol]
        List of docked RDKit molecule objects
    receptor_pdb : Path
        Path to receptor PDB file (predicted structure)
    work_dir : Path
        Working directory for Boltz-2 (contains manifest.json and constraints)
    output_dir : Path
        Output directory for Boltzina results
    fname : str
        Base filename for output files
    input_ligand_name : str
        Name of ligand in input files
    batch_size : int
        Batch size for Boltz-2 scoring
    num_workers : int
        Number of workers for parallel processing
    seed : Optional[int]
        Random seed
    prepared_mols_dict : Optional[dict]
        Dictionary of prepared molecules (optional)

    Returns
    -------
    list[tuple[float, float]]
        List of (affinity_pred_value1, affinity_probability_binary1) tuples
        Returns (0.0, 0.0) for molecules that fail
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up debug logger
    logger = setup_debug_logger(output_dir)
    logger.info("Starting Boltzina scoring")
    logger.info(f"Processing {len(docked_mols)} molecules")
    
    # Convert docked molecules to PDB format
    temp_pdb_dir = output_dir / "temp_pdb"
    temp_pdb_dir.mkdir(parents=True, exist_ok=True)
    
    ligand_pdb_files = []
    prepared_mols_dict = {} if prepared_mols_dict is None else prepared_mols_dict.copy()
    
    for i, mol in enumerate(docked_mols):
        if mol is None:
            ligand_pdb_files.append(None)
            continue

        ligand_pdb = temp_pdb_dir / f"ligand_{i}.pdb"
        success = sdf_to_pdb_with_naming(mol, ligand_pdb, mol_name="MOL")
        
        if success and ligand_pdb.exists():
            ligand_pdb_files.append(ligand_pdb)
            
            # Reload molecule from PDB to ensure PDB residue info is set
            mol_from_pdb = Chem.MolFromPDBFile(str(ligand_pdb), sanitize=False)
            if mol_from_pdb is not None:
                # Ensure all atoms have names set
                for atom in mol_from_pdb.GetAtoms():
                    if not atom.HasProp("name"):
                        info = atom.GetPDBResidueInfo()
                        if info and info.GetName():
                            atom.SetProp("name", info.GetName().strip().upper())
                        else:
                            atom.SetProp("name", f"{atom.GetSymbol().upper()}{atom.GetIdx() + 1}")
                
                prepared_mols_dict[ligand_pdb.stem] = mol_from_pdb
            else:
                prepared_mols_dict[ligand_pdb.stem] = mol
        else:
            ligand_pdb_files.append(None)
    
    logger.info(f"Converted {len([f for f in ligand_pdb_files if f is not None])}/{len(docked_mols)} molecules to PDB")

    # Pre-process receptor: remove chain B once (not per-ligand)
    receptor_no_b_path = output_dir / "receptor_no_B.pdb"
    receptor_no_b_path = _remove_chain_b_from_receptor(receptor_pdb, receptor_no_b_path)
    
    # Create receptor-ligand complex files (parallelized)
    ligand_indices = []  # Store original molecule indices
    boltz_ligand_idx_to_mol_idx = {}  # Map Boltzina's ligand_idx to original molecule index
    
    # Prepare tasks for parallel processing
    complex_tasks = []
    for i, ligand_pdb in enumerate(ligand_pdb_files):
        if ligand_pdb is None:
            continue
        
        ligand_output_dir = output_dir / "out" / str(len(complex_tasks))
        ligand_output_dir.mkdir(parents=True, exist_ok=True)
        base_name = ligand_pdb.stem
        
        complex_tasks.append({
            "index": i,
            "ligand_pdb": ligand_pdb,
            "ligand_output_dir": ligand_output_dir,
            "base_name": base_name,
            "receptor_no_b": receptor_no_b_path,
        })
    
    # Create complexes (parallelized if num_workers > 1)
    max_workers = min(num_workers, len(complex_tasks)) if num_workers > 1 and len(complex_tasks) > 1 else 1
    boltz_idx = 0
    
    if max_workers > 1:
        # Parallel processing
        def _create_complex_task(task_data):
            """Helper function for parallel complex creation."""
            try:
                complex_file = create_receptor_ligand_complex(
                    receptor_pdb,
                    task_data["ligand_pdb"],
                    task_data["ligand_output_dir"],
                    task_data["base_name"],
                    input_ligand_name="MOL",
                    base_ligand_name="MOL",
                    receptor_no_b=task_data["receptor_no_b"],
                )
                return task_data["index"], complex_file
            except Exception:
                return task_data["index"], None
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_create_complex_task, task): task for task in complex_tasks}
            for future in as_completed(futures):
                mol_idx, complex_file = future.result()
                if complex_file is not None and complex_file.exists() and complex_file.stat().st_size > 0:
                    if validate_complex_cif_has_ligand(complex_file, "MOL"):
                        ligand_indices.append(mol_idx)
                        boltz_ligand_idx_to_mol_idx[boltz_idx] = mol_idx
                        boltz_idx += 1
    else:
        # Sequential processing
        for task in complex_tasks:
            complex_file = create_receptor_ligand_complex(
                receptor_pdb,
                task["ligand_pdb"],
                task["ligand_output_dir"],
                task["base_name"],
                input_ligand_name="MOL",
                base_ligand_name="MOL",
                receptor_no_b=receptor_no_b_path,
            )
            
            if complex_file is not None and complex_file.exists() and complex_file.stat().st_size > 0:
                if validate_complex_cif_has_ligand(complex_file, "MOL"):
                    ligand_indices.append(task["index"])
                    boltz_ligand_idx_to_mol_idx[boltz_idx] = task["index"]
                    boltz_idx += 1

    if not ligand_indices:
        logger.error("All molecules failed complex creation. Returning zero scores.")
        return [(0.0, 0.0) for _ in docked_mols]
    
    logger.info(f"Created complexes for {len(ligand_indices)}/{len(docked_mols)} ligands")

    # Generate minimal config
    config_path = generate_minimal_boltzina_config(
        work_dir, receptor_pdb, output_dir, fname, input_ligand_name
    )
    
    # Read the config to get the actual fname that was used (may have been updated from manifest)
    with open(config_path, "r") as f:
        boltzina_config = json.load(f)
    actual_fname = boltzina_config["fname"]
    logger.debug(f"Actual fname from manifest.json: {actual_fname}")

    # Prepare mols_dict for Boltzina (more robust than PDB files)
    # Update prepared_mols_dict keys to match actual PDB file names
    # The keys need to match ligand_path.stem in Boltzina
    # Filter to only include ligands that successfully created complexes
    updated_prepared_mols_dict = {}
    for boltz_idx in sorted(boltz_ligand_idx_to_mol_idx.keys()):
        mol_idx = boltz_ligand_idx_to_mol_idx[boltz_idx]
        ligand_pdb = ligand_pdb_files[mol_idx]
        if ligand_pdb is not None:
            # Key should match ligand_path.stem (PDB filename without extension)
            ligand_stem = ligand_pdb.stem
            if ligand_stem in prepared_mols_dict:
                updated_prepared_mols_dict[ligand_stem] = prepared_mols_dict[ligand_stem]
    
    # Get ligand file paths for scoring
    # Note: Boltzina's run_scoring_only expects ligand PDB files
    # It will call _process_pose for each, but we've already created complexes
    # So we pass the ligand PDB files - Boltzina will check if complexes exist
    # Create mapping from Boltzina's ligand_idx to our original molecule index
    # Only include ligands that successfully created complexes
    ligand_files = []
    # Also create a mapping from ligand_path.stem to the molecule for Boltzina
    # Boltzina uses ligand_path.stem as the key in mol_dict
    ligand_stem_to_mol = {}
    for boltz_idx in sorted(boltz_ligand_idx_to_mol_idx.keys()):
        mol_idx = boltz_ligand_idx_to_mol_idx[boltz_idx]
        ligand_pdb = ligand_pdb_files[mol_idx]
        if ligand_pdb is not None:
            # Convert to absolute path to avoid issues with relative paths
            ligand_path_resolved = Path(ligand_pdb).resolve()
            ligand_files.append(str(ligand_path_resolved))
            # Map ligand_path.stem to the molecule
            # This is what Boltzina will use as the key in mol_dict
            ligand_stem = ligand_path_resolved.stem
            if ligand_stem in updated_prepared_mols_dict:
                ligand_stem_to_mol[ligand_stem] = updated_prepared_mols_dict[ligand_stem]
            else:
                # Fallback: use the original key from prepared_mols_dict
                # Find matching key by checking if stem matches
                for key, mol in prepared_mols_dict.items():
                    if key == ligand_stem or key.endswith(f"_{boltz_idx}"):
                        ligand_stem_to_mol[ligand_stem] = mol
                        break
    
    # Update prepared_mols_dict to use ligand_path.stem as keys
    # This ensures Boltzina can find the molecules correctly
    prepared_mols_dict = ligand_stem_to_mol
    
    prepared_mols_file = output_dir / "prepared_mols.pkl"
    with open(prepared_mols_file, "wb") as f:
        pickle.dump(prepared_mols_dict, f)
    logger.debug(f"Saved prepared_mols_dict ({len(prepared_mols_dict)} molecules) to: {prepared_mols_file}")
    logger.debug(f"Prepared mols dict keys: {list(prepared_mols_dict.keys())}")

    if not ligand_files:
        logger.error("No valid ligand files for Boltzina scoring")
        return [(0.0, 0.0) for _ in docked_mols]

    logger.info(f"Running Boltzina scoring for {len(ligand_files)} ligands (out of {len(docked_mols)} total)")
    logger.debug(f"Ligand files: {ligand_files}")

    # Run Boltzina scoring in boltzina conda environment using wrapper script
    try:
        # Create config for wrapper script
        # Convert all paths to absolute paths to avoid issues with relative paths
        wrapper_config = {
            "receptor_pdb": str(receptor_pdb.resolve()),
            "output_dir": str(output_dir.resolve()),
            "config": str(config_path.resolve()),
            "work_dir": str(work_dir.resolve()),
            "seed": seed,
            "num_workers": num_workers,
            "batch_size": batch_size,
            "prepared_mols_file": str(prepared_mols_file.resolve()) if prepared_mols_file.exists() else None,
            "input_ligand_name": "MOL",  # Use "MOL" because that's what we wrote in the PDB files, and Boltzina skips UNL
            "base_ligand_name": "MOL",  # Use "MOL" instead of "UNL" because Boltzina's parse_mmcif skips UNL ligands
            "fname": actual_fname,  # Use fname from manifest.json
            "ligand_files": [str(Path(f).resolve()) for f in ligand_files],  # Convert all to absolute paths
        }

        wrapper_config_path = output_dir / "boltzina_wrapper_config.json"
        wrapper_config_path = wrapper_config_path.resolve()  # Ensure absolute path
        with open(wrapper_config_path, "w") as f:
            json.dump(wrapper_config, f, indent=2)

        # Run wrapper script in boltzina conda environment
        wrapper_script = Path(__file__).parent / "boltzina_scoring_wrapper.py"
        
        try:
            result = run_python_in_conda_env(
                str(wrapper_script.resolve()),  # Ensure absolute path
                "boltzina",
                args=[str(wrapper_config_path)],  # Already absolute
                cwd=str(output_dir.resolve()),  # Ensure cwd is also absolute
                capture_output=True,  # Capture output to see errors
            )
            logger.info("Boltzina scoring completed successfully")
            if result.stdout:
                logger.debug(f"Boltzina stdout: {result.stdout}")
            if result.stderr:
                logger.debug(f"Boltzina stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running Boltzina scoring wrapper: {e}")
            logger.error(f"Command: {e.cmd}")
            logger.error(f"Return code: {e.returncode}")
            if hasattr(e, 'stdout') and e.stdout:
                logger.error(f"Boltzina stdout: {e.stdout}")
            if hasattr(e, 'stderr') and e.stderr:
                logger.error(f"Boltzina stderr: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Error running Boltzina scoring wrapper: {e}")
            raise
        
        # Post-process manifest to add ligand chain and affinity information (if missing)
        # This is a fallback - normally the decoy ligand chain B added during prediction
        # ensures the manifest includes it, but this code ensures it's present regardless
        manifest_path = output_dir / "boltz_out" / "processed" / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            
            # Update each record to include ligand chain B and affinity info
            updated = False
            for record in manifest.get("records", []):
                # Check if ligand chain already exists
                has_ligand_chain = any(
                    chain.get("chain_name") == "B" or chain.get("mol_type") == 3
                    for chain in record.get("chains", [])
                )
                
                if not has_ligand_chain:
                    # Add ligand chain B
                    max_chain_id = max([chain.get("chain_id", -1) for chain in record.get("chains", [])], default=-1)
                    max_entity_id = max([chain.get("entity_id", -1) for chain in record.get("chains", [])], default=-1)
                    
                    ligand_chain = {
                        "chain_id": max_chain_id + 1,
                        "chain_name": "B",
                        "mol_type": 3,  # NONPOLYMER
                        "cluster_id": -1,
                        "msa_id": -1,
                        "num_residues": 1,  # Ligand is typically a single residue
                        "valid": True,
                        "entity_id": max_entity_id + 1
                    }
                    record["chains"].append(ligand_chain)
                    
                    # Update num_chains in structure
                    if "structure" in record:
                        record["structure"]["num_chains"] = len(record["chains"])
                    updated = True
                
                # Set affinity information to specify ligand chain
                ligand_chain_id = next(
                    (chain.get("chain_id") for chain in record.get("chains", [])
                     if chain.get("chain_name") == "B" or chain.get("mol_type") == 3),
                    None
                )
                if ligand_chain_id is not None and record.get("affinity") is None:
                    record["affinity"] = {"chain_id": ligand_chain_id}
                    updated = True
            
            # Save updated manifest if changes were made
            if updated:
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=4)
                logger.info("Updated manifest to include ligand chain B and affinity information")
        
        # Extract results - need to map Boltzina's ligand_idx back to original molecule indices
        # Boltzina uses: f"{fname}_{ligand_idx}_{pose_idx}"
        # where ligand_idx is the enumerate index from run_scoring_only (0, 1, 2...)
        # Note: fname is now the actual record ID from manifest.json
        logger.debug("Extracting results from Boltzina output...")
        results = [(0.0, 0.0) for _ in docked_mols]  # Initialize all to (0.0, 0.0)
        
        pose_idx = "1"  # We only use one pose
        for boltz_ligand_idx, mol_idx in boltz_ligand_idx_to_mol_idx.items():
            fname_result = f"{actual_fname}_{boltz_ligand_idx}_{pose_idx}"
            affinity_file = output_dir / "boltz_out" / "predictions" / fname_result / f"affinity_{fname_result}.json"

            if affinity_file.exists():
                try:
                    with open(affinity_file, "r") as f:
                        affinity_data = json.load(f)

                    affinity_value1 = affinity_data.get("affinity_pred_value1", 0.0)
                    affinity_prob1 = affinity_data.get("affinity_probability_binary1", 0.0)
                    results[mol_idx] = (float(affinity_value1), float(affinity_prob1))
                    logger.debug(f"Ligand {mol_idx} (boltz_idx {boltz_ligand_idx}): affinity={affinity_value1}, prob={affinity_prob1}")
                except Exception as e:
                    logger.error(f"Error parsing affinity file {affinity_file}: {e}")
                    results[mol_idx] = (0.0, 0.0)
            else:
                logger.warning(f"Affinity file not found: {affinity_file}")
                results[mol_idx] = (0.0, 0.0)
        
        logger.info(f"Successfully extracted results for {len([r for r in results if r != (0.0, 0.0)])} ligands")
        logger.info("=" * 80)
        logger.info("Boltzina scoring completed")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("=" * 80)
        return results

    except Exception as e:
        logger.error(f"Error running Boltzina scoring: {e}", exc_info=True)
        return [(0.0, 0.0) for _ in docked_mols]

