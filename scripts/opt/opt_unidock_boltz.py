import argparse
import datetime
import os
import sqlite3
import sys
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).parent.parent.parent
src_dir = script_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
from omegaconf import DictConfig, OmegaConf

from synthflow.config import Config, init_empty


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

    # docking (not used for co-folding, but kept for compatibility)
    parser.add_argument("--protein_path", type=str, help="Path to the protein structure file (not used in co-folding).")
    parser.add_argument("--center", type=float, nargs=3, help="Center coordinates (x y z) for search box (not used in co-folding).")
    parser.add_argument(
        "--ref_ligand_path", type=str, help="Path to the reference ligand file (not used in co-folding)."
    )
    parser.add_argument("--size", type=float, nargs=3, help="Size (x y z) of the search box (not used in co-folding).")

    # pose prediction
    parser.add_argument("--pose_model", type=str, help="Path to the pose prediction model checkpoint.")
    parser.add_argument("--pose_steps", type=int, help="Number of steps for pose prediction.")

    # boltz setup
    parser.add_argument("--boltz_base_yaml", type=str, help="Path to Boltz-2 base.yaml config file (contains protein sequence). Required.")
    parser.add_argument("--boltz_msa_path", type=str, help="Path to MSA file for Boltz-2 (optional, can be specified in base.yaml).")
    parser.add_argument("--boltz_cache_dir", type=str, help="Cache directory for Boltz-2 (default: ~/project/boltz_cache).")
    parser.add_argument("--boltz_use_msa_server", action="store_true", help="Use MSA server for Boltz-2 (default: False).")
    parser.add_argument("--boltz_target_residues", type=str, nargs="+", help="Target residues for pocket constraints (format: 'A:123'). Optional.")
    parser.add_argument("--boltz_reward_cache_path", type=str, help="Path to reward cache database (default: {result_dir}/{time}/boltz_reward_cache.db).")
    
    # resume training
    parser.add_argument("--resume_from", type=str, help="Path to checkpoint file to resume training from (e.g., model_state_200.pt).")
    parser.add_argument(
        "--resume_oracle_idx", 
        type=int, 
        help="Override starting oracle_idx when resuming. If not provided, will scan directories and prompt if mismatch detected."
    )

    args = parser.parse_args()

    # NOTE: override config
    param: DictConfig = OmegaConf.load(args.config)
    for key in vars(args):
        if key == "config":
            continue
        value = getattr(args, key)
        if value is not None:
            # Set nested keys properly
            if key.startswith("boltz_"):
                # Handle boltz nested config
                nested_key = key.replace("boltz_", "")
                if "boltz" not in param:
                    param["boltz"] = {}
                param["boltz"][nested_key] = value
            else:
                param[key] = value
    return param


def get_max_oracle_idx(log_dir: Path) -> int:
    """Find the maximum oracle_idx from existing boltz_cofold and pose directories."""
    max_idx = 0
    for subdir in ["boltz_cofold", "pose"]:
        dir_path = log_dir / subdir
        if dir_path.exists():
            for item in dir_path.iterdir():
                if item.name.startswith("oracle"):
                    try:
                        # Handle both "oracle123" and "oracle123_something"
                        idx_str = item.name.replace("oracle", "").split("_")[0]
                        idx = int(idx_str)
                        max_idx = max(max_idx, idx)
                    except ValueError:
                        continue
    return max_idx


def detect_data_loss_offset(log_dir: Path) -> int | None:
    """
    Analyze train.log to detect data loss by finding iteration jumps or multiple 'Starting training' entries.
    
    Returns the estimated number of oracles lost, or None if no data loss detected.
    """
    log_file = log_dir / "train.log"
    if not log_file.exists():
        return None
    
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
        
        # Look for multiple "Starting training" entries (indicates resume with data loss)
        starting_training_indices = [i for i, line in enumerate(lines) if "Starting training" in line]
        
        if len(starting_training_indices) <= 1:
            return None  # No data loss detected
        
        # Get the last "Starting training" index
        last_start_idx = starting_training_indices[-1]
        
        # Find the iteration number just before the last "Starting training"
        last_iter_before_resume = None
        for i in range(last_start_idx - 1, -1, -1):
            line = lines[i]
            if "iteration" in line and ":" in line:
                try:
                    parts = line.split("iteration")[1].split(":")[0].strip()
                    last_iter_before_resume = int(parts)
                    break
                except (ValueError, IndexError):
                    continue
        
        if last_iter_before_resume is None:
            return None
        
        # The offset is the last iteration number before the data loss
        # For example: if iteration 200 was the last before resume, 200 oracles were lost
        return last_iter_before_resume
        
    except Exception as e:
        print(f"Warning: Could not analyze train.log for data loss: {e}")
        return None


def validate_oracle_idx(log_dir: Path, checkpoint_step: int, user_override: int | None) -> int:
    """
    Validate and determine the correct starting oracle_idx.
    
    Returns the oracle_idx to use (next iteration will be oracle_idx + 1).
    """
    scanned_max = get_max_oracle_idx(log_dir)
    
    # If user provided override, use it
    if user_override is not None:
        print(f"Using user-provided oracle_idx: {user_override}")
        return user_override - 1  # Return idx so next oracle is user_override
    
    # Check for mismatch
    if scanned_max != checkpoint_step:
        print("\n" + "="*60)
        print("WARNING: Oracle index mismatch detected!")
        print("="*60)
        print(f"  Checkpoint step:        {checkpoint_step}")
        print(f"  Max oracle in dirs:     {scanned_max}")
        print()
        
        # Try to detect data loss offset from train.log
        data_loss_offset = detect_data_loss_offset(log_dir)
        
        if scanned_max > checkpoint_step:
            # Training ran past checkpoint, need to backup extra oracles
            print("Training continued past the checkpoint before stopping.")
            print(f"  Oracles {checkpoint_step + 1} to {scanned_max} exist and will be backed up.")
            print()
            print("Options:")
            print(f"  - Use {checkpoint_step + 1} to resume from checkpoint (backs up oracles {checkpoint_step + 1}-{scanned_max})")
            print(f"  - Use {scanned_max + 1} to continue from existing data (no backup needed)")
            default = checkpoint_step + 1
        else:
            # Previous data loss
            print("This mismatch may indicate previous data loss.")
            
            if data_loss_offset is not None:
                # Infer the correct starting point
                corrected_idx = checkpoint_step + 1
                print(f"  Detected data loss offset from train.log: {data_loss_offset} oracles")
                print(f"  Max oracle in dirs ({scanned_max}) + offset ({data_loss_offset}) = {scanned_max + data_loss_offset}")
                print()
                print("Suggested action:")
                print(f"  - Use {corrected_idx} (recommended: resume from checkpoint, accounting for data loss)")
                default = corrected_idx
            else:
                print("Options:")
                print(f"  - Use {scanned_max + 1} to continue from existing data")
                print(f"  - Use {checkpoint_step + 1} to match checkpoint step")
                default = scanned_max + 1
        
        print()
        
        while True:
            try:
                user_input = input(f"Enter starting oracle_idx [{default}]: ").strip()
                if user_input == "":
                    return default - 1
                return int(user_input) - 1  # Return the idx that will produce next oracle
            except ValueError:
                print("Please enter a valid integer.")
    
    return scanned_max


def backup_oracles_if_needed(log_dir: Path, resume_oracle_idx: int, scanned_max: int):
    """
    Backup oracles that would be overwritten when resuming.
    
    If resume_oracle_idx <= scanned_max, oracles from resume_oracle_idx to scanned_max
    will be moved to a timestamped backup folder.
    """
    if resume_oracle_idx > scanned_max:
        return  # No backup needed
    
    # Create timestamped backup folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = log_dir / f"oracle_backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nBacking up oracles {resume_oracle_idx} to {scanned_max}...")
    print(f"Backup location: {backup_dir}")
    
    backed_up_count = 0
    for subdir in ["boltz_cofold", "pose"]:
        src_dir = log_dir / subdir
        if not src_dir.exists():
            continue
        
        backup_subdir = backup_dir / subdir
        backup_subdir.mkdir(parents=True, exist_ok=True)
        
        for item in src_dir.iterdir():
            if item.name.startswith("oracle"):
                try:
                    # Extract oracle index from name
                    idx_str = item.name.replace("oracle", "").split("_")[0]
                    idx = int(idx_str)
                    
                    # Backup if it would be overwritten
                    if resume_oracle_idx <= idx <= scanned_max:
                        dest = backup_subdir / item.name
                        import shutil
                        shutil.move(str(item), str(dest))
                        backed_up_count += 1
                except ValueError:
                    continue
    
    print(f"Backed up {backed_up_count} oracle directories to {backup_dir}")


if __name__ == "__main__":
    from tasks.unidock_boltz import UniDockBoltzMOOTrainer

    param = parse_args()

    config = init_empty(Config())

    config.desc = "Multi objective optimization for UniDock Boltz co-folding"
    # Default objectives - can be overridden in config file
    # QED is now optional, lilly filter is applied to boltz reward calculation
    # Set default objectives if not specified in config
    try:
        objectives = OmegaConf.select(config, "task.moo.objectives", default=None)
        if objectives is None or (isinstance(objectives, list) and len(objectives) == 0):
            config.task.moo.objectives = ["boltz"]  # Default to just boltz (which includes lilly filter)
    except (AttributeError, KeyError):
        config.task.moo.objectives = ["boltz"]  # Default to just boltz (which includes lilly filter)
    config.print_every = 10
    config.checkpoint_every = 100
    config.store_all_checkpoints = True
    config.num_workers_retrosynthesis = 4

    # Handle resume from checkpoint
    resume_from = OmegaConf.select(param, "resume_from", default=None)
    if resume_from:
        checkpoint_path = Path(resume_from).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Set log_dir to the directory containing the checkpoint
        log_dir = checkpoint_path.parent
        config.log_dir = str(log_dir)
        print(f"Using existing log directory: {config.log_dir}")
        
        # Extract checkpoint step from filename (e.g., model_state_1500.pt -> 1500)
        checkpoint_step = int(checkpoint_path.stem.split("_")[-1])
        
        # Get max oracle in directories
        scanned_max = get_max_oracle_idx(log_dir)
        
        # Validate oracle_idx (will prompt user if mismatch detected)
        user_override = OmegaConf.select(param, "resume_oracle_idx", default=None)
        resume_oracle_idx = validate_oracle_idx(log_dir, checkpoint_step, user_override)
        
        # Backup oracles that would be overwritten
        backup_oracles_if_needed(log_dir, resume_oracle_idx + 1, scanned_max)
        
        # Set resume configuration
        config.start_at_step = checkpoint_step
        config.pretrained_model_path = str(checkpoint_path)
        
        # CRITICAL: Never delete on resume!
        config.overwrite_existing_exp = False
        
        # Store resume info for trainer
        resume_mode = True
        resume_oracle_idx_value = resume_oracle_idx + 1  # Next oracle to create
        
        print(f"Resuming from checkpoint step {checkpoint_step}, starting oracle_idx at {resume_oracle_idx_value}")
    else:
        time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        config.log_dir = os.path.join(param.result_dir, time)
        
        # Set overwrite_existing_exp=True to allow overwriting if directory exists
        config.overwrite_existing_exp = True
        config.start_at_step = 0
        
        # Not resuming
        resume_mode = False
        resume_oracle_idx_value = None

    # Validate boltz configuration
    boltz_base_yaml = OmegaConf.select(param, "boltz.base_yaml", default=None)
    if not boltz_base_yaml:
        raise ValueError("boltz.base_yaml is required in config file. This file contains the protein sequence for Boltz-2 co-folding.")
    
    # Calculate center from target_residues if provided, otherwise use ref_ligand_path or center from config
    center = None
    ref_ligand_path = param.ref_ligand_path if param.ref_ligand_path else None
    
    target_residues = OmegaConf.select(param, "boltz.target_residues", default=None)
    if target_residues and param.protein_path:
        # Calculate center from target residues (similar to boltzina setup)
        from synthflow.utils.boltzina_setup import generate_grid_config
        try:
            grid_config = generate_grid_config(
                pdb_path=param.protein_path,
                residues=target_residues,
            )
            grid_dims = grid_config["grid_dimensions"]
            center = (
                grid_dims["center_x"],
                grid_dims["center_y"],
                grid_dims["center_z"],
            )
            print(f"Calculated grid center from target_residues: {center}")
        except Exception as e:
            print(f"Warning: Could not calculate center from target_residues: {e}")
            print("Falling back to ref_ligand_path or center from config")
            center = None
    
    # Use center from config if not calculated from residues
    if center is None:
        if param.center is not None:
            center = tuple(param.center)
        elif ref_ligand_path:
            # Center will be calculated from ref_ligand_path in set_pocket
            center = None
        else:
            raise ValueError(
                "Either center, ref_ligand_path, or boltz.target_residues must be provided "
                "to define the binding site for molecule generation."
            )
    
    # generative environment
    config.env_dir = param.env_dir
    config.algo.action_subsampling.sampling_ratio = param.subsampling_ratio
    config.algo.max_nodes = param.max_atoms

    # Set boltz paths before trainer initialization
    config.task.boltz.base_yaml = str(Path(boltz_base_yaml).resolve())
    if OmegaConf.select(param, "boltz.msa_path", default=None) is not None:
        config.task.boltz.msa_path = str(Path(param.boltz.msa_path).resolve())
    if OmegaConf.select(param, "boltz.cache_dir", default=None) is not None:
        config.task.boltz.cache_dir = param.boltz.cache_dir
    if OmegaConf.select(param, "boltz.use_msa_server", default=None) is not None:
        config.task.boltz.use_msa_server = param.boltz.use_msa_server
    if OmegaConf.select(param, "boltz.target_residues", default=None) is not None:
        config.task.boltz.target_residues = param.boltz.target_residues
    if OmegaConf.select(param, "boltz.reward_cache_path", default=None) is not None:
        config.task.boltz.reward_cache_path = str(Path(param.boltz.reward_cache_path).resolve())

    # Docking config (needed for pocket context setup - defines where to generate molecules)
    # We need a valid protein_path and either center or ref_ligand_path
    if not param.protein_path:
        raise ValueError("protein_path is required in config file for pocket context setup (defines where to generate molecules)")
    config.task.docking.protein_path = str(Path(param.protein_path).resolve())
    config.task.docking.center = center  # Calculated from target_residues or from config
    config.task.docking.size = [16, 16, 16]  # Size for pocket extraction
    config.task.docking.ff_opt = "uff"
    config.task.docking.ref_ligand_path = str(Path(ref_ligand_path).resolve()) if ref_ligand_path else None

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

    # Set module-level resume flags before instantiation
    import tasks.unidock_boltz as unidock_boltz_module
    unidock_boltz_module._RESUME_MODE = resume_mode
    unidock_boltz_module._RESUME_ORACLE_IDX = resume_oracle_idx_value
    
    # Initialize trainer (no database restoration needed - we never deleted anything!)
    trainer = UniDockBoltzMOOTrainer(config)
    
    print(f"Using Boltz co-folding with base YAML: {boltz_base_yaml}")
    print(f"Each sampled ligand will be co-folded with the protein using Boltz-2")
    
    trainer.run()

