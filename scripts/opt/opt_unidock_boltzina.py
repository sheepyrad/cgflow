import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).parent.parent.parent
src_dir = script_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from omegaconf import DictConfig, OmegaConf

from synthflow.config import Config, init_empty
from synthflow.utils.boltzina_setup import setup_boltzina_environment


def parse_args() -> DictConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file.")

    # config override
    # save dir
    parser.add_argument("--result_dir", type=str, help="Directory to save results.")

    # environment
    parser.add_argument("--env_dir", type=str, help="Directory containing environment data.")
    parser.add_argument("--max_atoms", type=int, help="Number of maximum atoms.")
    parser.add_argument(
        "--subsampling_ratio", type=float, help="Subsampling ratio for action space; Memory-variance trade-off."
    )

    # opt
    parser.add_argument("--num_steps", type=int, help="Number of training steps for optimization.")
    parser.add_argument("--num_sampling_per_step", type=int, help="Number of sampling for each train step.")
    parser.add_argument(
        "--temperature",
        type=int,
        nargs="+",
        help="Temperature (Exploration-Exploitation Trade-off). e.g., `--temp 32` (const); `--temp 1 64` (unif)",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")

    # docking
    parser.add_argument("--protein_path", type=str, help="Path to the protein structure file.")
    parser.add_argument("--center", type=float, nargs=3, help="Center coordinates (x y z) for search box.")
    parser.add_argument(
        "--ref_ligand_path", type=str, help="Path to the reference ligand file (required if center is not given)."
    )
    parser.add_argument("--size", type=float, nargs=3, help="Size (x y z) of the search box.")

    # pose prediction
    parser.add_argument("--pose_model", type=str, help="Path to the pose prediction model checkpoint.")
    parser.add_argument("--pose_steps", type=int, help="Number of steps for pose prediction.")

    # boltzina setup
    parser.add_argument("--boltzina_base_yaml", type=str, help="Path to Boltz-2 base.yaml config file (contains protein sequence). Can be specified in config file.")
    parser.add_argument("--boltzina_ref_ligand", type=str, help="Path to reference ligand for validation docking.")
    parser.add_argument("--boltzina_target_residues", type=str, nargs="+", help="Target residues for grid box generation (format: 'A:123'). Required in config file.")
    parser.add_argument("--boltzina_work_dir", type=str, help="Working directory for Boltz-2 (auto-generated if not provided).")
    # Deprecated: protein_pdb is not used - structure is predicted from base.yaml
    parser.add_argument("--boltzina_protein_pdb", type=str, help="(DEPRECATED) Not used - structure predicted from base.yaml")

    args = parser.parse_args()

    # NOTE: override config
    param: DictConfig = OmegaConf.load(args.config)
    for key in vars(args):
        if key == "config":
            continue
        value = getattr(args, key)
        if value is not None:
            # Set nested keys properly using OmegaConf
            if key.startswith("boltzina_"):
                # Handle boltzina nested config
                nested_key = key.replace("boltzina_", "")
                # Special handling for target_residues - keep as list
                if nested_key == "target_residues":
                    OmegaConf.set(param, f"boltzina.{nested_key}", value)
                else:
                    OmegaConf.set(param, f"boltzina.{nested_key}", value)
            else:
                OmegaConf.set(param, key, value)
    return param


if __name__ == "__main__":
    from tasks.unidock_boltzina import UniDockBoltzinaMOOTrainer

    param = parse_args()

    config = init_empty(Config())

    config.desc = "Multi objective optimization for UniDock Boltzina with QED"
    config.task.moo.objectives = ["boltz", "qed"]
    config.print_every = 10
    config.num_workers_retrosynthesis = 4

    time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    config.log_dir = os.path.join(param.result_dir, time)
    
    # Set overwrite_existing_exp=True to allow overwriting if directory exists
    config.overwrite_existing_exp = True

    # Pre-training setup: Predict structure and setup Boltzina environment
    # This creates subdirectories in config.log_dir. If the directory exists from a previous run,
    # the trainer will delete it and recreate it (due to overwrite_existing_exp=True).
    print("Setting up Boltzina environment...")
    target_residues = OmegaConf.select(param, "boltzina.target_residues", default=None)
    if not target_residues:
        raise ValueError("--boltzina_target_residues is required. Please provide target residues (e.g., --boltzina_target_residues A:123 B:456)")
    
    boltzina_base_yaml = OmegaConf.select(param, "boltzina.base_yaml", default=None)
    if not boltzina_base_yaml:
        raise ValueError("boltzina.base_yaml is required in config file. This file contains the protein sequence for Boltz-2 structure prediction.")
    
    # ref_ligand_path is optional - grid center and size are calculated from target residues
    boltzina_ref_ligand = OmegaConf.select(param, "boltzina.ref_ligand", default=None)
    
    # Temporarily create log_dir for setup if it doesn't exist
    # The trainer will recreate it during initialization
    os.makedirs(config.log_dir, exist_ok=True)
    
    setup_info = setup_boltzina_environment(
        base_yaml=boltzina_base_yaml,
        ref_ligand_path=boltzina_ref_ligand,  # Optional - not used for docking
        result_dir=config.log_dir,
        target_residues=target_residues,
        protein_pdb=OmegaConf.select(param, "boltzina.protein_pdb", default=None),
        work_dir=OmegaConf.select(param, "boltzina.work_dir", default=None),
        use_msa_server=False,  # MSA is specified in base.yaml file
        seed=OmegaConf.select(param, "seed", default=1),
    )

    # generative environment
    config.env_dir = param.env_dir
    config.algo.action_subsampling.sampling_ratio = param.subsampling_ratio
    config.algo.max_nodes = param.max_atoms

    # Set docking/boltzina paths before trainer initialization (trainer requires them)
    # Copy receptor file and entire Boltz-2 work directory to locations outside log_dir
    # so they persist during trainer initialization (trainer will delete log_dir)
    
    # Copy receptor PDB file
    temp_receptor_path = Path(param.result_dir) / f"temp_receptor_{time}.pdb"
    original_receptor_path = Path(setup_info.get("receptor_copy") or setup_info["receptor_pdb"]).resolve()
    if original_receptor_path.exists():
        shutil.copy2(original_receptor_path, temp_receptor_path)
        print(f"Copied receptor to temporary location: {temp_receptor_path}")
    else:
        # Fallback to actual receptor_pdb path
        temp_receptor_path = Path(setup_info["receptor_pdb"]).resolve()
        if not temp_receptor_path.exists():
            raise FileNotFoundError(f"Receptor PDB not found: {original_receptor_path} or {temp_receptor_path}")
    
    # Copy entire Boltz-2 work directory (contains manifest.json and constraints for scoring)
    temp_work_dir = Path(param.result_dir) / f"temp_boltz_work_{time}"
    original_work_dir = Path(setup_info["work_dir"]).resolve()
    if original_work_dir.exists():
        if temp_work_dir.exists():
            shutil.rmtree(temp_work_dir)
        shutil.copytree(original_work_dir, temp_work_dir)
        print(f"Copied Boltz-2 work directory to temporary location: {temp_work_dir}")
    else:
        raise FileNotFoundError(f"Boltz-2 work directory not found: {original_work_dir}")
    
    config.task.docking.protein_path = str(temp_receptor_path.resolve())
    config.task.docking.center = setup_info["grid_center"]
    config.task.docking.size = [16, 16, 16]  # Cube of 16
    config.task.docking.ff_opt = "uff"
    config.task.docking.ref_ligand_path = None  # CGFlow uses center and size directly
    
    config.task.boltzina.receptor_pdb = str(temp_receptor_path.resolve())
    config.task.boltzina.work_dir = str(temp_work_dir.resolve())
    if OmegaConf.select(param, "boltzina.fname", default=None) is not None:
        config.task.boltzina.fname = param.boltzina.fname
    if OmegaConf.select(param, "boltzina.batch_size", default=None) is not None:
        config.task.boltzina.batch_size = param.boltzina.batch_size
    if OmegaConf.select(param, "boltzina.num_workers", default=None) is not None:
        config.task.boltzina.num_workers = param.boltzina.num_workers

    # opt
    config.num_training_steps = param.num_steps
    config.algo.num_from_policy = param.num_sampling_per_step
    config.seed = param.seed

    # temperature
    if len(param.temperature) == 1:
        config.cond.temperature.sample_dist = "constant"
    elif len(param.temperature) == 2:
        config.cond.temperature.sample_dist = "uniform"
    else:
        raise ValueError("Temperature should be a single value for constant or two values for uniform.")
    config.cond.temperature.dist_params = param.temperature

    # pose prediction
    config.cgflow.ckpt_path = param.pose_model
    config.cgflow.num_inference_steps = param.pose_steps

    # extras
    config.algo.sampling_tau = param.sampling_tau  # EMA
    config.algo.train_random_action_prob = param.random_action_prob  # suggest to set positive value

    config.replay.use = True
    config.replay.warmup = config.algo.num_from_policy * param.replay_warmup_step
    config.replay.capacity = param.replay_capacity
    config.replay.num_from_replay = config.algo.num_from_policy  # buffer sampling size

    # Initialize trainer - this will delete and recreate log_dir if overwrite_existing_exp=True
    # So we need to recreate the setup after trainer initialization
    trainer = UniDockBoltzinaMOOTrainer(config)
    
    # Recreate setup after trainer initialization (since trainer deleted the directory)
    # Pass the temporary work_dir so boltzina_setup.py can reuse the existing Boltz-2 predictions
    print("Recreating Boltzina setup after trainer initialization...")
    setup_info = setup_boltzina_environment(
        base_yaml=boltzina_base_yaml,
        ref_ligand_path=boltzina_ref_ligand,
        result_dir=config.log_dir,
        target_residues=target_residues,
        protein_pdb=OmegaConf.select(param, "boltzina.protein_pdb", default=None),
        work_dir=str(temp_work_dir.resolve()),  # Use temporary work_dir so it reuses existing predictions
        use_msa_server=False,
        seed=OmegaConf.select(param, "seed", default=1),
    )
    
    # Update task with setup results after trainer initialization
    # Resolve paths to absolute and verify files exist
    receptor_pdb_path = Path(setup_info.get("receptor_copy") or setup_info["receptor_pdb"]).resolve()
    
    # Verify file exists
    if not receptor_pdb_path.exists():
        raise FileNotFoundError(
            f"Receptor PDB file not found: {receptor_pdb_path}\n"
            f"This file should have been created during setup recreation."
        )
    
    # Update config first (for consistency)
    trainer.cfg.task.docking.protein_path = str(receptor_pdb_path)
    trainer.cfg.task.docking.center = setup_info["grid_center"]
    trainer.cfg.task.docking.size = [16, 16, 16]  # Cube of 16
    trainer.cfg.task.docking.ff_opt = "uff"
    trainer.cfg.task.docking.ref_ligand_path = None
    trainer.cfg.task.boltzina.receptor_pdb = str(receptor_pdb_path)
    trainer.cfg.task.boltzina.work_dir = str(Path(setup_info["work_dir"]).resolve())
    
    # Update task attributes directly (task reads from config in __init__, but we need to update after initialization)
    trainer.task.protein_path = receptor_pdb_path
    trainer.task.center = setup_info["grid_center"]
    trainer.task.size = [16, 16, 16]  # Cube of 16
    trainer.task.ff_opt = "uff"
    trainer.task.boltzina_receptor_pdb = receptor_pdb_path
    trainer.task.boltzina_work_dir = Path(setup_info["work_dir"]).resolve()
    
    # Update environment context with new receptor path
    trainer.ctx.set_pocket(
        str(receptor_pdb_path),
        center=setup_info["grid_center"],
        ref_ligand_path=None,
    )
    
    # Clean up temporary files
    # Remove temporary receptor file if it's different from the final path
    if temp_receptor_path.exists() and temp_receptor_path != receptor_pdb_path:
        try:
            temp_receptor_path.unlink()
            print(f"Removed temporary receptor file: {temp_receptor_path}")
        except Exception as e:
            print(f"Warning: Could not remove temporary receptor file {temp_receptor_path}: {e}")
    
    # Remove temporary work directory if it's different from the final path
    # (boltzina_setup.py should have copied it to boltzina_setup/boltz_work)
    final_work_dir = Path(setup_info["work_dir"]).resolve()
    if temp_work_dir.exists() and temp_work_dir != final_work_dir:
        try:
            shutil.rmtree(temp_work_dir)
            print(f"Removed temporary Boltz-2 work directory: {temp_work_dir}")
        except Exception as e:
            print(f"Warning: Could not remove temporary work directory {temp_work_dir}: {e}")
    
    print(f"Using grid center {setup_info['grid_center']} and size [16, 16, 16] (cube)")
    print(f"Updated task with receptor PDB path: {receptor_pdb_path}")
    
    trainer.run()

