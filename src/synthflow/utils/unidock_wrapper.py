"""Wrapper script to run UniDock docking in unidock-env conda environment.

This script is called as a subprocess from unidock.py and boltzina_setup.py
to ensure UniDock runs in the correct conda environment.

Uses UniDock CLI directly instead of Python API to avoid bugs.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol as RDMol


def run_etkdg(mol: RDMol, sdf_path: Path | str, seed: int = 1) -> bool:
    """Generate 3D conformers for RDKit molecules."""
    from rdkit.Chem import AllChem

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


def main():
    """Run UniDock docking with parameters from JSON."""
    if len(sys.argv) < 2:
        print("Usage: python unidock_wrapper.py <config_json>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"UniDock wrapper starting...")
    print(f"Config: {config_path}")
    print(f"Protein: {config['protein_path']}")
    print(f"Number of ligands: {len(config['input_sdf_files'])}")
    print(f"Grid center: {config['center']}")
    print(f"Grid size: {config['size']}")

    # Load molecules from SDF files
    input_sdf_files = [Path(f) for f in config["input_sdf_files"]]
    mols = []
    for sdf_file in input_sdf_files:
        try:
            mol = next(Chem.SDMolSupplier(str(sdf_file), sanitize=False))
            if mol is None:
                print(f"Warning: Failed to load molecule from {sdf_file}")
                continue
            mols.append(mol)
            print(f"Loaded molecule {len(mols)}: {sdf_file.name} ({mol.GetNumAtoms()} atoms)")
        except Exception as e:
            print(f"Error loading molecule from {sdf_file}: {e}")
            continue

    if len(mols) == 0:
        print("Error: No valid molecules loaded")
        sys.exit(1)

    # Prepare ligands
    import tempfile
    from pathlib import Path as PathType

    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = PathType(out_dir)
        sdf_list = []
        for i, mol in enumerate(mols):
            ligand_file = out_dir / f"{i}.sdf"
            flag = run_etkdg(mol, ligand_file, seed=config.get("seed", 1))
            if flag:
                sdf_list.append(ligand_file)
                print(f"Prepared ligand {len(sdf_list)}: {ligand_file.name}")
            else:
                print(f"Warning: Failed to prepare 3D conformer for molecule {i}")

        if len(sdf_list) == 0:
            print("Error: No ligands prepared successfully")
            sys.exit(1)

        print(f"Running UniDock docking on {len(sdf_list)} ligands...")
        try:
            # Resolve protein path to absolute path
            protein_path = PathType(config["protein_path"]).resolve()
            print(f"Using protein path: {protein_path}")
            
            # Prepare output directory
            savedir = out_dir / "savedir"
            savedir.mkdir(parents=True, exist_ok=True)
            
            # Build UniDock CLI command
            # Use --gpu_batch with SDF files for batch docking
            unidock_cmd = [
                "unidock",
                "--receptor", str(protein_path),
                "--gpu_batch"] + [str(sdf) for sdf in sdf_list] + [
                "--center_x", str(round(config["center"][0], 3)),
                "--center_y", str(round(config["center"][1], 3)),
                "--center_z", str(round(config["center"][2], 3)),
                "--size_x", str(round(config["size"][0], 3)),
                "--size_y", str(round(config["size"][1], 3)),
                "--size_z", str(round(config["size"][2], 3)),
                "--dir", str(savedir),
                "--search_mode", config.get("search_mode", "balance"),
                "--num_modes", "1",
                "--seed", str(config.get("seed", 1)),
                "--verbosity", "1",
            ]
            
            print(f"Running UniDock command: {' '.join(unidock_cmd)}")
            
            # Run UniDock CLI
            result = subprocess.run(
                unidock_cmd,
                capture_output=True,
                text=True,
                check=False,  # Don't raise on error, we'll handle it
            )
            
            # Log output
            if result.stdout:
                print("UniDock stdout:")
                print(result.stdout)
            if result.stderr:
                print("UniDock stderr:")
                print(result.stderr)
            
            if result.returncode != 0:
                raise RuntimeError(f"UniDock CLI failed with exit code {result.returncode}")
            
            print(f"UniDock docking completed")
        except Exception as e:
            print(f"Error during UniDock docking: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Extract results
        # UniDock CLI saves files with names based on input file names
        # Check savedir for output files
        savedir = out_dir / "savedir"
        results = []
        output_sdf_files = []
        
        # List all output files in savedir for debugging
        all_output_files = list(savedir.glob("*"))
        print(f"Output directory contains {len(all_output_files)} files: {[f.name for f in all_output_files]}")
        
        # UniDock CLI may create files with different naming conventions
        # Try to find output files matching input file names
        for i, sdf_file in enumerate(sdf_list):
            try:
                # UniDock CLI may create files with .pdbqt extension or keep .sdf
                # Try different possible output file names
                possible_names = [
                    savedir / f"{i}.sdf",
                    savedir / f"{i}.pdbqt",
                    savedir / f"{sdf_file.stem}.sdf",
                    savedir / f"{sdf_file.stem}.pdbqt",
                    savedir / f"{sdf_file.stem}_out.sdf",
                    savedir / f"{sdf_file.stem}_out.pdbqt",
                    savedir / f"{sdf_file.name}",
                    savedir / f"{sdf_file.stem}_dock.sdf",
                    savedir / f"{sdf_file.stem}_dock.pdbqt",
                ]
                
                docked_file = None
                for name in possible_names:
                    if name.exists():
                        docked_file = name
                        break
                
                if docked_file is None:
                    # List all files in savedir for debugging
                    print(f"Warning: Docked file not found for molecule {i} (input: {sdf_file.name}). Available files: {[f.name for f in all_output_files]}")
                    results.append((None, 0.0))
                    output_sdf_files.append(None)
                    continue
                
                print(f"Found docked file for molecule {i}: {docked_file.name}")
                
                # Try to load as SDF first
                docked_rdmol = None
                try:
                    docked_rdmol = next(Chem.SDMolSupplier(str(docked_file), sanitize=False))
                except:
                    # If SDF fails, might be PDBQT - try to convert or skip
                    print(f"Warning: Failed to parse {docked_file} as SDF (might be PDBQT)")
                
                if docked_rdmol is None:
                    print(f"Warning: Failed to parse docked molecule {i}")
                    results.append((None, 0.0))
                    output_sdf_files.append(None)
                    continue
                
                # Extract docking score from molecule properties
                docking_score = 0.0
                try:
                    if docked_rdmol.HasProp("docking_score"):
                        docking_score = float(docked_rdmol.GetProp("docking_score"))
                    elif docked_rdmol.HasProp("minimizedAffinity"):
                        docking_score = float(docked_rdmol.GetProp("minimizedAffinity"))
                    elif docked_rdmol.HasProp("rmsd"):
                        # Try to get score from comment or other properties
                        pass
                except:
                    print(f"Warning: Could not extract docking score for molecule {i}")
                
                print(f"Molecule {i}: docking score = {docking_score}")

                # Save docked molecule to output directory
                output_sdf = PathType(config["output_dir"]) / f"{i}.sdf"
                output_sdf.parent.mkdir(parents=True, exist_ok=True)
                with Chem.SDWriter(str(output_sdf)) as w:
                    w.write(docked_rdmol)
                output_sdf_files.append(str(output_sdf))
                results.append((str(output_sdf), docking_score))
            except Exception as e:
                print(f"Error processing molecule {i}: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                results.append((None, 0.0))
                output_sdf_files.append(None)

        # Save results
        results_file = PathType(config["output_dir"]) / "unidock_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(
                {
                    "output_sdf_files": output_sdf_files,
                    "docking_scores": [score for _, score in results],
                },
                f,
                indent=2,
            )

        print(f"UniDock docking completed. Results saved to {config['output_dir']}")
        print(f"Successfully docked {sum(1 for f in output_sdf_files if f is not None)}/{len(mols)} molecules")


if __name__ == "__main__":
    main()

