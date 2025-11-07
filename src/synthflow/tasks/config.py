from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class MOOTaskConfig:
    """Common Config for the MOOTasks

    Attributes
    ----------
    objectives : list[str]
        The objectives to use for the multi-objective optimization.
    n_valid : int
        The number of valid cond_info tensors to sample.
    n_valid_repeats : int
        The number of times to repeat the valid cond_info tensors.
    online_pareto_front : bool
        Whether to calculate the pareto front online.
    """

    objectives: list[str] = field(default_factory=lambda: [])
    n_valid: int = 15
    n_valid_repeats: int = 128
    log_topk: bool = False
    online_pareto_front: bool = True


@dataclass
class DockingTaskConfig:
    """Config for DockingTask

    Attributes
    ----------
    protein_path: str (path)
        Protein path
    center: tuple[float, float, float]
        Pocket center
    ref_ligand_path: str (path)
        Reference ligand path
    size: tuple[float, float, float]
        Search box size
    redocking: bool
        if True, run docking; otherwise, run local opt
    ff_opt: str
        When redocking is False, run FF optimization before local opt
    """

    protein_path: str = MISSING
    center: tuple[float, float, float] | None = None
    size: tuple[float, float, float] = (30, 30, 30)
    ref_ligand_path: str | None = None
    redocking: bool = True  # TODO: change to docking mode (score-only, local-opt, redock)
    ff_opt: str = "mmff"  # 'none', 'uff', 'mmff'
    exhaustiveness: int = 8  # Exhaustiveness for Vina docking


@dataclass
class ConstraintConfig:
    """Config for Filtering

    Attributes
    ----------
    rule: str (path)
        DrugFilter Rule
            - None
            - lipinski
            - veber
    """

    rule: str | None = None


@dataclass
class PocketConditionalConfig:
    """Config for PocketConditional Training

    Attributes
    ----------
    proxy: tuple[str, str, str] (proxy_name, docking_program, train_dataset)
        Proxy Key from PharmacoNet
    """

    proxy: tuple[str, str, str] = ("TacoGFN_Reward", "QVina", "ZINCDock15M")
    protein_dir: str = "./data/experiments/CrossDocked2020/"
    train_key: str = "./data/experiments/CrossDocked2020/train_keys.csv"


@dataclass
class BoltzinaTaskConfig:
    """Config for Boltzina Task

    Attributes
    ----------
    receptor_pdb: str (path)
        Path to predicted receptor PDB file (from Boltz-2)
    work_dir: str (path)
        Working directory for Boltz-2 (contains manifest.json and constraints)
    fname: str
        Base filename for Boltzina output files
    batch_size: int
        Batch size for Boltz-2 scoring
    num_workers: int
        Number of workers for parallel processing
    """

    receptor_pdb: str = MISSING
    work_dir: str = MISSING
    fname: str = "cgflow_ligand"
    batch_size: int = 1
    num_workers: int = 1


@dataclass
class BoltzTaskConfig:
    """Config for Boltz Co-folding Task

    Attributes
    ----------
    base_yaml: str (path)
        Path to Boltz-2 base.yaml config file (contains protein sequence)
    msa_path: str (path) | None
        Path to MSA file for Boltz-2 (optional, can be specified in base.yaml)
    cache_dir: str (path) | None
        Cache directory for Boltz-2 (default: ~/project/boltz_cache)
    use_msa_server: bool
        Whether to use MSA server for Boltz-2 (default: False)
    target_residues: list[str] | None
        List of target residues for pocket constraints (format: ['A:123', 'B:456'])
    """

    base_yaml: str = MISSING
    msa_path: str | None = None
    cache_dir: str | None = None
    use_msa_server: bool = False
    target_residues: list[str] | None = None


@dataclass
class TasksConfig:
    moo: MOOTaskConfig = field(default_factory=MOOTaskConfig)
    constraint: ConstraintConfig = field(default_factory=ConstraintConfig)
    docking: DockingTaskConfig = field(default_factory=DockingTaskConfig)
    pocket_conditional: PocketConditionalConfig = field(default_factory=PocketConditionalConfig)
    boltzina: BoltzinaTaskConfig = field(default_factory=BoltzinaTaskConfig)
    boltz: BoltzTaskConfig = field(default_factory=BoltzTaskConfig)
