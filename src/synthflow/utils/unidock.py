import subprocess
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Mol as RDMol

from synthflow.utils.conda_env import run_in_conda_env, run_python_in_conda_env


def prepare_protein_pdbqt(receptor_pdb: Path, output_pdbqt: Path) -> bool:
    """
    Prepare protein structure using unidocktools proteinprep.
    
    Converts PDB to PDBQT format suitable for UniDock docking.
    
    Args:
        receptor_pdb: Path to input PDB file
        output_pdbqt: Path to output PDBQT file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
        
        # Run proteinprep command in unidock-env conda environment
        command = [
            "unidocktools", "proteinprep",
            "-r", str(receptor_pdb.resolve()),
            "-o", str(output_pdbqt.resolve()),
            "-ph",  # Add hydrogens
        ]
        
        result = run_in_conda_env(
            command,
            conda_env="unidock-env",
            cwd=output_pdbqt.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0 and output_pdbqt.exists():
            return True
        else:
            return False
            
    except subprocess.CalledProcessError as e:
        return False
    except Exception as e:
        return False


def run_etkdg(mol: RDMol, sdf_path: Path | str, seed: int = 1) -> bool:
    if mol.GetNumAtoms() == 0:
        return False
    try:
        param = AllChem.srETKDGv3()
        param.randomSeed = seed
        mol = Chem.Mol(mol)
        mol.RemoveAllConformers()
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, param)
        mol = Chem.RemoveHs(mol)
        assert mol.GetNumConformers() > 0
        with Chem.SDWriter(str(sdf_path)) as w:
            w.write(mol)
    except Exception:
        return False
    else:
        return True


def docking(
    rdmols: list[RDMol],
    protein_path: str | Path,
    center: tuple[float, float, float],
    seed: int = 1,
    size: float | tuple[float, float, float] = 20.0,
    search_mode: str = "balance",
):
    """
    Run UniDock docking in unidock-env conda environment.

    This function ensures UniDock runs in the correct conda environment
    by using the Python API via conda environment execution.

    Parameters
    ----------
    rdmols : list[RDMol]
        List of RDKit molecules to dock
    protein_path : str | Path
        Path to protein PDB file
    center : tuple[float, float, float]
        Grid center coordinates
    seed : int
        Random seed for docking
    size : float | tuple[float, float, float]
        Grid size (single value or tuple for x, y, z)
    search_mode : str
        Search mode: "fast", "balance", or "detail"

    Returns
    -------
    list[tuple[RDMol | None, float]]
        List of (docked_molecule, docking_score) tuples
    """
    if isinstance(size, float | int):
        size = (size, size, size)

    protein_path = Path(protein_path)

    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        
        # Prepare receptor: convert PDB to PDBQT using unidocktools proteinprep
        # This is required for UniDock docking
        receptor_pdbqt = out_dir / "receptor.pdbqt"
        if not prepare_protein_pdbqt(protein_path, receptor_pdbqt):
            # If preparation fails, try using PDB directly (might fail)
            receptor_to_use = protein_path
        else:
            receptor_to_use = receptor_pdbqt
        
        # Prepare ligands
        sdf_list = []
        for i, mol in enumerate(rdmols):
            ligand_file = out_dir / f"{i}.sdf"
            flag = run_etkdg(mol, ligand_file, seed=seed)
            if flag:
                sdf_list.append(ligand_file)

        if len(sdf_list) == 0:
            return [(None, 0.0) for _ in rdmols]

        # Run UniDock using CLI (avoids Python API bugs)
        savedir = out_dir / "savedir"
        savedir.mkdir(parents=True, exist_ok=True)
        
        # Build UniDock CLI command
        unidock_cmd = [
            "unidock",
            "--receptor", str(receptor_to_use.resolve()),
            "--gpu_batch",
        ] + [str(sdf) for sdf in sdf_list] + [
            "--center_x", str(round(center[0], 3)),
            "--center_y", str(round(center[1], 3)),
            "--center_z", str(round(center[2], 3)),
            "--size_x", str(round(size[0], 3)),
            "--size_y", str(round(size[1], 3)),
            "--size_z", str(round(size[2], 3)),
            "--dir", str(savedir),
            "--search_mode", search_mode,
            "--num_modes", "1",
            "--seed", str(seed),
            "--verbosity", "1",
        ]
        
        try:
            result = run_in_conda_env(
                unidock_cmd,
                conda_env="unidock-env",
                cwd=out_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            return [(None, 0.0) for _ in rdmols]
        except Exception as e:
            return [(None, 0.0) for _ in rdmols]

        # Load results
        # UniDock CLI may create files with different naming conventions
        savedir = out_dir / "savedir"
        res: list[tuple[None, float] | tuple[RDMol, float]] = []
        for i in range(len(rdmols)):
            docked_rdmol = None
            docking_score = 0.0
            
            # Try different possible output file names
            sdf_file = sdf_list[i] if i < len(sdf_list) else None
            possible_names = [
                savedir / f"{i}.sdf",
                savedir / f"{i}.pdbqt",
            ]
            if sdf_file:
                possible_names.extend([
                    savedir / f"{sdf_file.stem}.sdf",
                    savedir / f"{sdf_file.stem}.pdbqt",
                    savedir / f"{sdf_file.stem}_out.sdf",
                    savedir / f"{sdf_file.stem}_out.pdbqt",
                    savedir / f"{sdf_file.name}",
                    savedir / f"{sdf_file.stem}_dock.sdf",
                    savedir / f"{sdf_file.stem}_dock.pdbqt",
                ])
            
            docked_file = None
            for name in possible_names:
                if name.exists():
                    docked_file = name
                    break
            
            if docked_file:
                try:
                    # Try to load as SDF
                    docked_rdmol = next(Chem.SDMolSupplier(str(docked_file), sanitize=False))
                    if docked_rdmol is not None:
                        # Extract docking score
                        if docked_rdmol.HasProp("docking_score"):
                            docking_score = float(docked_rdmol.GetProp("docking_score"))
                        elif docked_rdmol.HasProp("minimizedAffinity"):
                            docking_score = float(docked_rdmol.GetProp("minimizedAffinity"))
                except Exception:
                    # If parsing fails, try to find any SDF file in savedir
                    all_sdf_files = list(savedir.glob("*.sdf"))
                    if all_sdf_files and i < len(all_sdf_files):
                        try:
                            docked_rdmol = next(Chem.SDMolSupplier(str(all_sdf_files[i]), sanitize=False))
                            if docked_rdmol is not None and docked_rdmol.HasProp("docking_score"):
                                docking_score = float(docked_rdmol.GetProp("docking_score"))
                        except Exception:
                            pass
            
            res.append((docked_rdmol, docking_score))

        return res


def scoring(
    rdmols: list[RDMol],
    protein_path: str | Path,
    center: tuple[float, float, float],
    size: float = 25.0,
):
    """Run UniDock scoring-only mode in unidock-env conda environment."""
    protein_path = Path(protein_path)

    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        
        # Prepare receptor: convert PDB to PDBQT using unidocktools proteinprep
        receptor_pdbqt = out_dir / "receptor.pdbqt"
        if not prepare_protein_pdbqt(protein_path, receptor_pdbqt):
            receptor_to_use = protein_path
        else:
            receptor_to_use = receptor_pdbqt
        
        sdf_list = []
        for i, mol in enumerate(rdmols):
            ligand_file = out_dir / f"{i}.sdf"
            try:
                lig_pos = np.array(mol.GetConformer().GetPositions())
                min_x, min_y, min_z = lig_pos.min(0)
                assert center[0] - size / 2 < min_x
                assert center[1] - size / 2 < min_y
                assert center[2] - size / 2 < min_z
                max_x, max_y, max_z = lig_pos.max(0)
                assert center[0] + size / 2 > max_x
                assert center[1] + size / 2 > max_y
                assert center[2] + size / 2 > max_z
                with Chem.SDWriter(str(ligand_file)) as w:
                    w.write(mol)
            except Exception:
                pass
            else:
                sdf_list.append(ligand_file)
        
        if len(sdf_list) > 0:
            # Run UniDock scoring using CLI
            savedir = out_dir / "savedir"
            savedir.mkdir(parents=True, exist_ok=True)
            
            unidock_cmd = [
                "unidock",
                "--receptor", str(receptor_to_use.resolve()),
                "--gpu_batch",
            ] + [str(sdf) for sdf in sdf_list] + [
                "--center_x", str(round(center[0], 3)),
                "--center_y", str(round(center[1], 3)),
                "--center_z", str(round(center[2], 3)),
                "--size_x", str(round(size, 3)),
                "--size_y", str(round(size, 3)),
                "--size_z", str(round(size, 3)),
                "--dir", str(savedir),
                "--score_only",
                "--verbosity", "1",
            ]
            
            try:
                run_in_conda_env(
                    unidock_cmd,
                    conda_env="unidock-env",
                    cwd=out_dir,
                    check=False,  # Don't raise on error
                    capture_output=True,
                    text=True,
                )
            except Exception:
                pass

        # Load results (similar to docking function)
        savedir = out_dir / "savedir"
        res: list[tuple[None, float] | tuple[RDMol, float]] = []
        for i in range(len(rdmols)):
            docked_rdmol = None
            docking_score = 0.0
            
            # Try different possible output file names
            sdf_file = sdf_list[i] if i < len(sdf_list) else None
            possible_names = [
                savedir / f"{i}.sdf",
                savedir / f"{i}.pdbqt",
            ]
            if sdf_file:
                possible_names.extend([
                    savedir / f"{sdf_file.stem}.sdf",
                    savedir / f"{sdf_file.stem}.pdbqt",
                    savedir / f"{sdf_file.stem}_out.sdf",
                    savedir / f"{sdf_file.stem}_out.pdbqt",
                    savedir / f"{sdf_file.name}",
                ])
            
            docked_file = None
            for name in possible_names:
                if name.exists():
                    docked_file = name
                    break
            
            if docked_file:
                try:
                    docked_rdmol = next(Chem.SDMolSupplier(str(docked_file), sanitize=False))
                    if docked_rdmol is not None:
                        if docked_rdmol.HasProp("docking_score"):
                            docking_score = float(docked_rdmol.GetProp("docking_score"))
                        elif docked_rdmol.HasProp("minimizedAffinity"):
                            docking_score = float(docked_rdmol.GetProp("minimizedAffinity"))
                except Exception:
                    # Try to find any SDF file in savedir
                    all_sdf_files = list(savedir.glob("*.sdf"))
                    if all_sdf_files and i < len(all_sdf_files):
                        try:
                            docked_rdmol = next(Chem.SDMolSupplier(str(all_sdf_files[i]), sanitize=False))
                            if docked_rdmol is not None and docked_rdmol.HasProp("docking_score"):
                                docking_score = float(docked_rdmol.GetProp("docking_score"))
                        except Exception:
                            pass
            
            res.append((docked_rdmol, docking_score))
        return res
