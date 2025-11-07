import argparse
import datetime
import os
import sys
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).parent.parent.parent
src_dir = script_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

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
    config.num_workers_retrosynthesis = 4

    time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    config.log_dir = os.path.join(param.result_dir, time)
    
    # Set overwrite_existing_exp=True to allow overwriting if directory exists
    config.overwrite_existing_exp = True

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

    # Initialize trainer
    trainer = UniDockBoltzMOOTrainer(config)
    
    print(f"Using Boltz co-folding with base YAML: {boltz_base_yaml}")
    print(f"Each sampled ligand will be co-folded with the protein using Boltz-2")
    
    trainer.run()

