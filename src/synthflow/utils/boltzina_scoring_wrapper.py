"""Wrapper script to run Boltzina scoring in boltzina conda environment.

This script is called as a subprocess from boltzina_integration.py
to ensure it runs in the correct conda environment.
"""

import json
import pickle
import sys
import traceback
from pathlib import Path

def log(message: str, file=sys.stderr):
    """Log message to stderr (captured by parent process)."""
    print(message, file=file, flush=True)

# Add boltzina to path
# Wrapper is at cgflow/src/synthflow/utils/boltzina_scoring_wrapper.py
# Boltzina is at workspace_root/boltzina
boltzina_path = Path(__file__).parent.parent.parent.parent.parent / "boltzina"
if str(boltzina_path) not in sys.path:
    sys.path.insert(0, str(boltzina_path))

log(f"Boltzina wrapper starting...")
log(f"Python path: {sys.executable}")
log(f"Boltzina path: {boltzina_path}")
log(f"Boltzina path exists: {boltzina_path.exists()}")

try:
    from boltzina_main import Boltzina
    log("Successfully imported Boltzina")
except Exception as e:
    log(f"ERROR: Failed to import Boltzina: {e}")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)


def main():
    """Run Boltzina scoring with parameters from JSON."""
    if len(sys.argv) < 2:
        log("ERROR: Usage: python boltzina_scoring_wrapper.py <config_json>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    # Resolve to absolute path to avoid issues with relative paths
    config_path = config_path.resolve()
    log(f"Reading config from: {config_path}")
    
    if not config_path.exists():
        log(f"ERROR: Config file does not exist: {config_path}")
        log(f"Current working directory: {Path.cwd()}")
        sys.exit(1)
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        log(f"Successfully loaded config with keys: {list(config.keys())}")
    except Exception as e:
        log(f"ERROR: Failed to load config: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # Verify required paths exist
    receptor_pdb = Path(config["receptor_pdb"])
    # Resolve to absolute path
    receptor_pdb = receptor_pdb.resolve()
    log(f"Receptor PDB: {receptor_pdb} (exists: {receptor_pdb.exists()})")
    if not receptor_pdb.exists():
        log(f"ERROR: Receptor PDB file does not exist: {receptor_pdb}")
        sys.exit(1)
    
    output_dir = Path(config["output_dir"])
    # Resolve to absolute path
    output_dir = output_dir.resolve()
    log(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    work_dir = config.get("work_dir")
    if work_dir:
        work_dir = Path(work_dir)
        # Resolve to absolute path
        work_dir = work_dir.resolve()
        log(f"Work directory: {work_dir} (exists: {work_dir.exists()})")
        manifest_path = work_dir / "processed" / "manifest.json"
        log(f"Manifest path: {manifest_path} (exists: {manifest_path.exists()})")
        if not manifest_path.exists():
            log(f"WARNING: Manifest.json not found at: {manifest_path}")
    
    config_file = Path(config["config"])
    # Resolve to absolute path
    config_file = config_file.resolve()
    log(f"Config file: {config_file} (exists: {config_file.exists()})")
    
    ligand_files = config["ligand_files"]
    log(f"Number of ligand files: {len(ligand_files)}")
    # Resolve all ligand file paths to absolute
    ligand_files = [str(Path(f).resolve()) for f in ligand_files]
    for i, ligand_file in enumerate(ligand_files[:5]):  # Log first 5
        ligand_path = Path(ligand_file)
        log(f"  Ligand {i}: {ligand_path} (exists: {ligand_path.exists()})")
    if len(ligand_files) > 5:
        log(f"  ... and {len(ligand_files) - 5} more")

    try:
        log("Initializing Boltzina...")
        # Initialize Boltzina
        boltzina = Boltzina(
            receptor_pdb=str(receptor_pdb.resolve()),
            output_dir=str(output_dir.resolve()),
            config=str(config_file.resolve()),
            work_dir=str(work_dir.resolve()) if work_dir else None,
            seed=config.get("seed"),
            num_workers=config.get("num_workers", 1),
            batch_size=config.get("batch_size", 1),
            num_boltz_poses=1,
            scoring_only=True,
            skip_run_structure=True,
            prepared_mols_file=config.get("prepared_mols_file"),
            input_ligand_name=config.get("input_ligand_name", "UNL"),
            base_ligand_name=config.get("base_ligand_name", "UNL"),
            fname=config.get("fname", "cgflow_ligand"),
        )
        log("Boltzina initialized successfully")
    except Exception as e:
        log(f"ERROR: Failed to initialize Boltzina: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    try:
        log("Running Boltzina scoring...")
        # Run scoring
        boltzina.run_scoring_only(ligand_files)
        log("Boltzina scoring completed successfully")
    except Exception as e:
        log(f"ERROR: Boltzina scoring failed: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # Save results summary
    try:
        results_summary = {
            "status": "success",
            "output_dir": str(output_dir),
            "num_ligands": len(ligand_files),
        }

        summary_path = output_dir / "boltzina_scoring_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        log(f"Results summary saved to: {summary_path}")
    except Exception as e:
        log(f"WARNING: Failed to save results summary: {e}")

    log(f"Boltzina scoring completed. Results saved to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: Unexpected error in main: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

