from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import threading
from pathlib import Path
import json
import subprocess
import yaml

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED
import medchem as mc

from gflownet.utils import sascore
from gflownet.utils.sqlite_log import BoltzinaSQLiteLogHook

from synthflow.config import Config
from synthflow.pocket_specific.trainer import RxnFlow3DTrainer_single
from synthflow.utils.boltz_reward_cache import BoltzRewardCache

from .docking import BaseDockingTask


# Module-level variables for resume mode (set by opt_unidock_boltz.py before trainer instantiation)
_RESUME_MODE: bool = False
_RESUME_ORACLE_IDX: int | None = None


class UniDockBoltzTask(BaseDockingTask):
    """Task for co-folding ligand and protein using Boltz-2."""
    
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # Get Boltz-specific config
        self.boltz_base_yaml = Path(cfg.task.boltz.base_yaml)
        self.boltz_msa_path = getattr(cfg.task.boltz, "msa_path", None)
        self.boltz_output_dir = Path(cfg.log_dir) / "boltz_cofold"
        self.boltz_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set oracle_idx from module-level variable if resuming
        if _RESUME_ORACLE_IDX is not None:
            self.oracle_idx = _RESUME_ORACLE_IDX
            print(f"Resuming with oracle_idx starting at: {self.oracle_idx}")
        self.boltz_cache_dir = getattr(cfg.task.boltz, "cache_dir", "~/project/boltz_cache")
        self.boltz_use_msa_server = getattr(cfg.task.boltz, "use_msa_server", False)
        self.boltz_target_residues = getattr(cfg.task.boltz, "target_residues", None)
        self.boltz_worker = int(getattr(cfg.task.boltz, "worker", 1))
        if self.boltz_worker < 1:
            raise ValueError(f"boltz.worker must be >= 1, got {self.boltz_worker}")
        self._cache_worker_idx_by_thread: dict[int, int] = {}
        self._cache_worker_idx_lock = threading.Lock()
        
        # Initialize reward cache
        cache_path = getattr(cfg.task.boltz, "reward_cache_path", None)
        if cache_path is None:
            cache_path = Path(cfg.log_dir) / "boltz_reward_cache.db"
        else:
            cache_path = Path(cache_path)
        self.reward_cache = BoltzRewardCache(str(cache_path))
        print(f"Initialized Boltz reward cache at: {cache_path}")
        print(f"Cache size: {self.reward_cache.get_db_size(fast=True):,} entries")
        
        # Storage for database logging
        self.batch_docking_scores = []  # Empty list - no docking in co-folding
        self.batch_boltz_scores = []  # List of dicts with all boltz scores for current batch
        self.batch_smiles = []  # List of SMILES for current batch
        self.batch_iteration = None  # Current training iteration
        
        # Load protein sequence from base_yaml
        with open(self.boltz_base_yaml, "r") as f:
            yaml_data = yaml.safe_load(f)
        
        # Extract protein sequence
        self.protein_sequence = None
        if "sequences" in yaml_data:
            for seq in yaml_data["sequences"]:
                if "protein" in seq:
                    protein_data = seq["protein"]
                    if isinstance(protein_data, dict):
                        self.protein_sequence = protein_data.get("sequence")
                    break
        
        if self.protein_sequence is None:
            raise ValueError(f"Could not find protein sequence in {self.boltz_base_yaml}")
        
        # Extract MSA path if available
        if self.boltz_msa_path is None and "sequences" in yaml_data:
            for seq in yaml_data["sequences"]:
                if "protein" in seq:
                    protein_data = seq["protein"]
                    if isinstance(protein_data, dict) and "msa" in protein_data:
                        self.boltz_msa_path = protein_data["msa"]
                        break
        
        # Restore topn_affinity if resuming (do this after parent __init__ sets it up)
        if _RESUME_MODE:
            restored = self.restore_topn_from_cache()
            if restored:
                self.topn_affinity = restored

    def restore_topn_from_cache(self) -> OrderedDict[str, float] | None:
        """Load top N molecules from boltz_reward_cache.db on resume."""
        cache_path = Path(self.cfg.log_dir) / "boltz_reward_cache.db"
        if not cache_path.exists():
            return None
        
        try:
            import sqlite3
            conn = sqlite3.connect(str(cache_path))
            cursor = conn.cursor()
            # Table is named 'entries' in BoltzRewardCache
            cursor.execute("""
                SELECT smiles, reward FROM entries 
                ORDER BY reward DESC 
                LIMIT 1000
            """)
            results = cursor.fetchall()
            conn.close()
            
            topn = OrderedDict((smiles, reward) for smiles, reward in results)
            print(f"Restored {len(topn)} top molecules from reward cache")
            return topn
        except Exception as e:
            print(f"Warning: Failed to restore topn_affinity from cache: {e}")
            return None

    def calc_affinities(self, mols: list[Chem.Mol]) -> list[float]:
        """Co-fold ligand and protein using Boltz-2 for each molecule."""
        if self.redocking:
            return self.run_cofold_with_boltz(mols)
        else:
            raise NotImplementedError("Local optimization not implemented for UniDockBoltzTask")

    @staticmethod
    def _empty_boltz_result() -> dict[str, float]:
        return {
            "affinity_ensemble": 0.0,
            "probability_ensemble": 0.0,
            "affinity_model1": 0.0,
            "probability_model1": 0.0,
            "affinity_model2": 0.0,
            "probability_model2": 0.0,
        }

    def _evaluate_uncached_smiles(
        self,
        smiles: str,
        mol_idx: int,
        batch_output_dir: Path,
    ) -> tuple[str, float, dict[str, float], tuple[str, float, str]]:
        """Evaluate one uncached SMILES and return score/result/cache-entry tuple."""
        # Check Lilly filter FIRST (before expensive Boltz calculation)
        try:
            lily_mask = mc.functional.lilly_demerit_filter(
                mols=[Chem.MolFromSmiles(smiles)],
                n_jobs=-1,
                progress=False,
                return_idx=False,
            )
            assert len(lily_mask.shape) == 1, "Lilly mask should be a 1D array"
            assert lily_mask.shape[0] == 1, "Lilly mask should have only one element"
            lily_mask = float(lily_mask[0])
            assert lily_mask == 0.0 or lily_mask == 1.0, "Lilly mask should be 0.0 or 1.0"
        except Exception as e:
            print(f"Warning: Failed to compute lilly filter for {smiles}: {e}")
            lily_mask = 0.0

        if lily_mask == 0.0:
            result = self._empty_boltz_result()
            boltz_score = 0.0
            return smiles, boltz_score, result, (smiles, boltz_score, json.dumps(result))

        try:
            # Create temporary directory for this molecule
            mol_temp_dir = batch_output_dir / f"mol_{mol_idx}"
            mol_temp_dir.mkdir(parents=True, exist_ok=True)

            # Create YAML input file for Boltz-2
            query_name = f"mol_{mol_idx}"
            yaml_input_path = mol_temp_dir / f"{query_name}.yaml"
            self._prepare_boltz_yaml(smiles, yaml_input_path)

            # Run Boltz-2 prediction
            boltz_output_dir = mol_temp_dir / "boltz_output"
            self._run_boltz_prediction(yaml_input_path, boltz_output_dir)

            # Extract affinity scores
            result = self._extract_boltz_results(boltz_output_dir, query_name)
            if result is None:
                result = self._empty_boltz_result()
                boltz_score = 0.0
            else:
                affinity_value1 = result.get("affinity_model1", 0.0)
                affinity_prob1 = result.get("probability_model1", 0.0)
                normalized_aff = max(0.0, (affinity_value1 * -1 + 2.0) / 4.0)
                boltz_score = float(normalized_aff * affinity_prob1 * lily_mask)

            return smiles, boltz_score, result, (smiles, boltz_score, json.dumps(result))
        except Exception as e:
            print(f"Error in Boltz co-folding for molecule {mol_idx} (SMILES: {smiles}): {e}")
            result = self._empty_boltz_result()
            return smiles, 0.0, result, (smiles, 0.0, json.dumps(result))

    def run_cofold_with_boltz(self, mols: list[Chem.Mol]) -> list[float]:
        """Co-fold ligand and protein using Boltz-2 for each molecule."""
        # Initialize arrays for all molecules
        boltz_scores = [0.0] * len(mols)
        boltz_results_all = [None] * len(mols)
        smiles_list = [""] * len(mols)
        docking_scores = [0.0] * len(mols)  # Empty list - no docking in co-folding
        
        # Create temporary directory for this batch
        batch_output_dir = self.boltz_output_dir / f"oracle{self.oracle_idx}"
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        
        # First, get SMILES for all molecules and check cache
        smiles_to_compute = []
        smiles_to_indices = {}  # Map SMILES to list of indices (in case of duplicates)
        
        for mol_idx, mol in enumerate(mols):
            if mol is None:
                boltz_results_all[mol_idx] = {
                    "affinity_ensemble": 0.0,
                    "probability_ensemble": 0.0,
                    "affinity_model1": 0.0,
                    "probability_model1": 0.0,
                    "affinity_model2": 0.0,
                    "probability_model2": 0.0,
                }
                # smiles_list and docking_scores already initialized
                continue
            
            try:
                smiles = Chem.MolToSmiles(mol)
                smiles_list[mol_idx] = smiles
                
                if smiles not in smiles_to_indices:
                    smiles_to_indices[smiles] = []
                smiles_to_indices[smiles].append(mol_idx)
                smiles_to_compute.append(smiles)
            except Exception as e:
                print(f"Error processing molecule {mol_idx}: {e}")
                boltz_results_all[mol_idx] = self._empty_boltz_result()
                smiles_list[mol_idx] = Chem.MolToSmiles(mol) if mol is not None else ""
        
        # Check cache for all unique SMILES
        cached_results = self.reward_cache.get_hits(smiles_to_compute)
        uncached_smiles = [s for s in smiles_to_compute if s not in cached_results]
        
        print(f"Cache hit rate: {len(cached_results)}/{len(smiles_to_compute)} ({100*len(cached_results)/len(smiles_to_compute):.1f}%)")
        
        # Initialize results arrays (will be filled in order)
        results_dict = {}  # smiles -> (score, result_dict)
        
        # Process cached results
        for smiles, (reward, info_str) in cached_results.items():
            try:
                # Parse info string to reconstruct result dict
                if info_str:
                    info_dict = json.loads(info_str) if info_str else {}
                else:
                    info_dict = {}
                
                # Reconstruct result dict from cached info
                result = {
                    "affinity_ensemble": info_dict.get("affinity_ensemble", 0.0),
                    "probability_ensemble": info_dict.get("probability_ensemble", 0.0),
                    "affinity_model1": info_dict.get("affinity_model1", 0.0),
                    "probability_model1": info_dict.get("probability_model1", 0.0),
                    "affinity_model2": info_dict.get("affinity_model2", 0.0),
                    "probability_model2": info_dict.get("probability_model2", 0.0),
                    "oracle_idx": info_dict.get("oracle_idx", None),
                    "mol_idx": info_dict.get("mol_idx", None),
                }
                results_dict[smiles] = (reward, result)
            except Exception as e:
                print(f"Error parsing cached result for {smiles}: {e}")
                # If parsing fails, treat as uncached
                if smiles not in uncached_smiles:
                    uncached_smiles.append(smiles)
        
        # Process uncached molecules
        new_cache_entries = []
        if uncached_smiles:
            worker_count = min(self.boltz_worker, len(uncached_smiles))
            print(f"Running Boltz for {len(uncached_smiles)} uncached SMILES with {worker_count} worker(s)")
            worker_results: dict[str, tuple[float, dict[str, float], tuple[str, float, str]]] = {}

            if worker_count == 1:
                for smiles in uncached_smiles:
                    mol_idx = smiles_to_indices[smiles][0]  # Use first occurrence
                    _, boltz_score, result, cache_entry = self._evaluate_uncached_smiles(
                        smiles=smiles,
                        mol_idx=mol_idx,
                        batch_output_dir=batch_output_dir,
                    )
                    worker_results[smiles] = (boltz_score, result, cache_entry)
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_to_smiles = {
                        executor.submit(
                            self._evaluate_uncached_smiles,
                            smiles,
                            smiles_to_indices[smiles][0],  # Use first occurrence
                            batch_output_dir,
                        ): smiles
                        for smiles in uncached_smiles
                    }

                    for future in as_completed(future_to_smiles):
                        smiles = future_to_smiles[future]
                        try:
                            _, boltz_score, result, cache_entry = future.result()
                        except Exception as e:
                            print(f"Unhandled worker error for {smiles}: {e}")
                            result = self._empty_boltz_result()
                            boltz_score = 0.0
                            cache_entry = (smiles, boltz_score, json.dumps(result))
                        worker_results[smiles] = (boltz_score, result, cache_entry)

            # Merge in FIFO order of uncached_smiles for deterministic assembly.
            for smiles in uncached_smiles:
                if smiles in worker_results:
                    boltz_score, result, cache_entry = worker_results[smiles]
                else:
                    result = self._empty_boltz_result()
                    boltz_score = 0.0
                    cache_entry = (smiles, boltz_score, json.dumps(result))
                results_dict[smiles] = (boltz_score, result)
                new_cache_entries.append(cache_entry)
        
        # Store new entries in cache
        if new_cache_entries:
            try:
                self.reward_cache.insert_entries(new_cache_entries)
                print(f"Cached {len(new_cache_entries)} new entries. Cache size: {self.reward_cache.get_db_size(fast=True):,}")
            except Exception as e:
                print(f"Warning: Failed to update cache: {e}")
        
        # Fill results arrays in original order
        for mol_idx, smiles in enumerate(smiles_list):
            if smiles and smiles in results_dict:
                score, result = results_dict[smiles]
                boltz_scores[mol_idx] = score
                boltz_results_all[mol_idx] = result.copy()  # Make a copy to avoid reference issues
        
        # Store for database logging
        self.batch_docking_scores = docking_scores
        self.batch_boltz_scores = boltz_results_all
        self.batch_smiles = smiles_list
        
        return boltz_scores
    
    def _prepare_boltz_yaml(self, ligand_smiles: str, yaml_output_path: Path):
        """Prepare YAML input file for Boltz-2 co-folding.
        
        Format matches the reference YAML structure:
        - id should be a string "A", not a list ["A"]
        - MSA path should be absolute
        - constraints contacts should be in flow style: [[A, 16], [A, 67], ...]
        """
        yaml_dict = {
            "sequences": [
                {
                    "protein": {
                        "id": "A",  # String, not list
                        "sequence": self.protein_sequence,
                    }
                },
                {
                    "ligand": {
                        "id": "B",
                        "smiles": ligand_smiles
                    }
                },
            ],
            "properties": [{"affinity": {"binder": "B"}}],
        }
        
        # Add MSA if available (use absolute path)
        if self.boltz_msa_path:
            msa_path = Path(self.boltz_msa_path).expanduser().resolve()
            if msa_path.exists():
                yaml_dict["sequences"][0]["protein"]["msa"] = str(msa_path)
            else:
                print(f"Warning: MSA file not found: {msa_path}")
        
        # Add constraints if target_residues are provided
        if self.boltz_target_residues:
            # Convert target_residues from "A:123" format to [A, 123] format
            contacts = []
            for residue_spec in self.boltz_target_residues:
                residue_spec = residue_spec.strip()
                if ":" in residue_spec:
                    chain_id, res_id = residue_spec.split(":", 1)
                    chain_id = chain_id.strip()
                    res_id = res_id.strip()
                    # Convert to [chain_id, int(res_id)] format
                    contacts.append([chain_id, int(res_id)])
                else:
                    # If format is incorrect, skip with warning
                    print(f"Warning: Invalid residue format '{residue_spec}', expected 'A:123' format. Skipping.")
            
            if contacts:
                yaml_dict["constraints"] = [
                    {
                        "pocket": {
                            "binder": "B",
                            "contacts": contacts
                        }
                    }
                ]
        
        # Write YAML with proper formatting
        # Use flow_style for contacts list to get [[A, 16], [A, 67], ...] format
        # We need to customize the dumper to use flow style for nested lists (contacts)
        def represent_list(dumper, data):
            # Check if this is a list of lists with 2 elements each (contacts format: [[A, 16], [A, 67]])
            if data and isinstance(data[0], list) and len(data[0]) == 2:
                return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
        
        yaml.add_representer(list, represent_list, Dumper=yaml.SafeDumper)
        
        with open(yaml_output_path, "w") as f:
            yaml.dump(yaml_dict, f, Dumper=yaml.SafeDumper, default_flow_style=False, sort_keys=False)
    
    def _run_boltz_prediction(self, yaml_input_path: Path, output_dir: Path):
        """Run Boltz-2 prediction command directly (no conda subprocess).
        
        Note: boltz predict accepts a YAML file path directly.
        This requires Boltz to be installed in the trainer environment.
        """
        # Use absolute paths
        yaml_input_path = yaml_input_path.resolve()
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build boltz predict command
        cache_dir = self._get_worker_cache_dir()
        max_attempts = 2
        try:
            for attempt in range(1, max_attempts + 1):
                cmd = [
                    "boltz",
                    "predict",
                    str(yaml_input_path),
                    "--out_dir",
                    str(output_dir),
                    "--output_format",
                    "pdb",
                    "--use_potentials",
                ]

                # Add cache directory if specified
                if cache_dir is not None:
                    cmd.extend(["--cache", str(cache_dir)])

                # Add MSA server flag if requested
                if self.boltz_use_msa_server:
                    cmd.append("--use_msa_server")

                try:
                    # Set environment variables for GPU memory management
                    import os
                    env_vars = os.environ.copy()
                    env_vars.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

                    # Run boltz command directly (no conda wrapper)
                    result = subprocess.run(
                        cmd,
                        cwd=None,
                        check=True,
                        capture_output=True,
                        text=True,
                        env=env_vars,
                    )
                    if result.stdout:
                        print(f"Boltz-2 prediction stdout: {result.stdout[:500]}")  # Print first 500 chars
                    break
                except subprocess.CalledProcessError as e:
                    error_msg = e.stderr if e.stderr else (e.stdout if hasattr(e, "stdout") else str(e))
                    is_badzip = "BadZipFile" in error_msg
                    if is_badzip and attempt < max_attempts and cache_dir is not None:
                        # Corrupted npz in cache can happen when multiple boltz subprocesses write/read shared files.
                        # Reset this worker-local cache and retry once.
                        print(f"Boltz cache corruption detected in {cache_dir}. Rebuilding cache and retrying once.")
                        shutil.rmtree(cache_dir, ignore_errors=True)
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        continue
                    print(f"Boltz-2 prediction failed: {error_msg}")
                    raise
        finally:
            # Clean up GPU memory after prediction
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def _get_worker_cache_dir(self) -> Path | None:
        """Return a cache directory isolated per thread when using multiple boltz workers."""
        if not self.boltz_cache_dir:
            return None

        base_cache_dir = Path(self.boltz_cache_dir).expanduser().resolve()
        if self.boltz_worker <= 1:
            base_cache_dir.mkdir(parents=True, exist_ok=True)
            return base_cache_dir

        thread_id = threading.get_ident()
        with self._cache_worker_idx_lock:
            worker_idx = self._cache_worker_idx_by_thread.get(thread_id)
            if worker_idx is None:
                worker_idx = len(self._cache_worker_idx_by_thread)
                self._cache_worker_idx_by_thread[thread_id] = worker_idx

        worker_cache_dir = base_cache_dir / f"worker_{worker_idx}"
        worker_cache_dir.mkdir(parents=True, exist_ok=True)
        return worker_cache_dir
    
    def _extract_boltz_results(self, boltz_output_dir: Path, query_name: str) -> dict | None:
        """Extract affinity results from Boltz-2 output.
        
        Boltz-2 output structure can vary:
        - <out_dir>/predictions/<query_name>/affinity_<query_name>.json
        - <out_dir>/boltz_results_*/predictions/<query_name>/affinity_<query_name>.json
        - <out_dir>/predictions/<query_name>/affinity_*.json (any affinity file)
        """
        boltz_output_dir = boltz_output_dir.resolve()
        
        # Try multiple possible output structures
        possible_paths = []
        
        # Structure 1: <out_dir>/predictions/<query_name>/affinity_<query_name>.json
        possible_paths.append(boltz_output_dir / "predictions" / query_name / f"affinity_{query_name}.json")
        
        # Structure 2: <out_dir>/predictions/<query_name>/affinity_*.json (any affinity file)
        predictions_dir = boltz_output_dir / "predictions" / query_name
        if predictions_dir.exists():
            possible_paths.extend(predictions_dir.glob("affinity_*.json"))
        
        # Structure 3: <out_dir>/boltz_results_*/predictions/<query_name>/affinity_*.json
        boltz_results_dirs = list(boltz_output_dir.glob("boltz_results_*/predictions"))
        for br_dir in boltz_results_dirs:
            query_path = br_dir / query_name
            if query_path.exists():
                possible_paths.extend(query_path.glob("affinity_*.json"))
        
        # Structure 4: Search recursively for any affinity JSON files
        if not possible_paths:
            all_affinity_files = list(boltz_output_dir.rglob("affinity_*.json"))
            possible_paths.extend(all_affinity_files)
        
        # Find the first existing affinity file
        affinity_file = None
        for path in possible_paths:
            if path.exists():
                affinity_file = path
                break
        
        if affinity_file is None:
            print(f"No affinity JSON file found for query '{query_name}' in {boltz_output_dir}")
            print(f"Searched paths: {[str(p) for p in possible_paths[:5]]}")
            return None
        
        try:
            with open(affinity_file, "r") as f:
                affinity_data = json.load(f)
            
            # Extract all scores
            result = {
                "affinity_ensemble": float(affinity_data.get("affinity_pred_value", 0.0)),
                "probability_ensemble": float(affinity_data.get("affinity_probability_binary", 0.0)),
                "affinity_model1": float(affinity_data.get("affinity_pred_value1", 0.0)),
                "probability_model1": float(affinity_data.get("affinity_probability_binary1", 0.0)),
                "affinity_model2": float(affinity_data.get("affinity_pred_value2", 0.0)),
                "probability_model2": float(affinity_data.get("affinity_probability_binary2", 0.0)),
            }
            return result
        except Exception as e:
            print(f"Error extracting Boltz results from {affinity_file}: {e}")
            return None


class UniDockBoltzMOOTask(UniDockBoltzTask):
    avg_reward_info: OrderedDict[str, float]

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.objectives = self.cfg.task.moo.objectives
        assert set(self.objectives) <= {"boltz", "qed", "sa", "lilly"}, f"Invalid objectives: {set(self.objectives) - {'boltz', 'qed', 'sa', 'lilly'}}"
        # Replace "vina" with "boltz" in objectives if present
        if "vina" in self.objectives:
            self.objectives = ["boltz" if obj == "vina" else obj for obj in self.objectives]

    def compute_rewards(self, mols: list[Chem.Mol]) -> torch.Tensor:
        self.save_pose(mols)
        flat_r: list[torch.Tensor] = []
        self.avg_reward_info = OrderedDict()
        for prop in self.objectives:
            if prop == "boltz":
                fr = self.calc_affinity_reward(mols)
            elif prop == "qed":
                fr = self.calc_qed_reward(mols)
            elif prop == "sa":
                fr = self.calc_sa_reward(mols)
            elif prop == "lilly":
                fr = self.calc_lilly_reward(mols)
            else:
                raise NotImplementedError(f"Objective {prop} is not implemented")
            flat_r.append(fr)
            self.avg_reward_info[prop] = fr.mean().item()
        flat_rewards = torch.stack(flat_r, dim=1).prod(dim=1, keepdim=True)
        assert flat_rewards.shape[0] == len(mols)
        return flat_rewards

    def calc_qed_reward(self, mols: list[Chem.Mol]) -> torch.Tensor:
        def calc_score(mol: Chem.Mol) -> float:
            try:
                return QED.qed(mol)
            except Exception:
                return 0.0

        return torch.tensor([calc_score(mol) for mol in mols])

    def calc_sa_reward(self, mols: list[Chem.Mol]) -> torch.Tensor:
        def calc_score(mol: Chem.Mol) -> float:
            try:
                return (10 - sascore.calculateScore(mol)) / 9
            except Exception:
                return 0.0

        return torch.tensor([calc_score(mol) for mol in mols])

    def calc_lilly_reward(self, mols: list[Chem.Mol]) -> torch.Tensor:
        """Calculate Lilly medchem filter reward (0.0 or 1.0)."""
        def calc_score(mol: Chem.Mol) -> float:
            try:
                smiles = Chem.MolToSmiles(mol)
                lily_mask = mc.functional.lilly_demerit_filter(
                    mols=[Chem.MolFromSmiles(smiles)], 
                    n_jobs=-1, 
                    progress=False, 
                    return_idx=False
                )
                assert len(lily_mask.shape) == 1, "Lilly mask should be a 1D array"
                assert lily_mask.shape[0] == 1, "Lilly mask should have only one element"
                lily_mask = float(lily_mask[0])
                assert lily_mask == 0.0 or lily_mask == 1.0, "Lilly mask should be 0.0 or 1.0"
                return lily_mask
            except Exception:
                return 0.0

        return torch.tensor([calc_score(mol) for mol in mols])

    def calc_affinity_reward(self, mols: list[Chem.Mol]) -> torch.Tensor:
        """Calculate Boltz score reward - already in 0-1 range, no transformation needed."""
        affinities = self.calc_affinities(mols)
        self.batch_affinity = affinities
        self.update_storage(mols, affinities)
        # Boltz scores are already positive (0-1 range), just clip to minimum
        fr = torch.tensor(affinities, dtype=torch.float32)
        return fr.clip(min=1e-5)

    def update_storage(self, mols: list[Chem.Mol], scores: list[float]):
        def _filter(mol: Chem.Mol) -> bool:
            """Check the object passes a property filter"""
            return QED.qed(mol) > 0.5

        pass_idcs = [i for i, mol in enumerate(mols) if _filter(mol)]
        pass_mols = [mols[i] for i in pass_idcs]
        pass_scores = [scores[i] for i in pass_idcs]
        
        # Update parent storage
        smiles_list = [Chem.MolToSmiles(mol) for mol in pass_mols]
        self.topn_affinity.update(zip(smiles_list, pass_scores, strict=True))
        
        # Sort by score in DESCENDING order (higher boltz score = better)
        # Unlike docking scores which are negative (lower = better)
        topn = sorted(list(self.topn_affinity.items()), key=lambda v: v[1], reverse=True)[:1000]
        self.topn_affinity = OrderedDict(topn)


class UniDockBoltzMOOTrainer(RxnFlow3DTrainer_single[UniDockBoltzMOOTask]):
    def setup(self):
        """Override setup to skip directory deletion when resuming."""
        if _RESUME_MODE:
            # Resume mode: don't delete anything, just ensure directories exist
            import os
            from rdkit import RDLogger
            from gflownet.utils.misc import set_worker_rng_seed
            from rxnflow.utils.misc import set_worker_env
            
            os.makedirs(self.cfg.log_dir, exist_ok=True)
            RDLogger.DisableLog("rdApp.*")
            torch.manual_seed(self.cfg.seed + 42)
            torch.cuda.manual_seed(self.cfg.seed + 42)
            set_worker_rng_seed(self.cfg.seed)
            
            # Set num_objectives before setup (like parent does)
            self.cfg.cond.moo.num_objectives = len(self.cfg.task.moo.objectives)
            
            self.setup_env()
            self.setup_data()
            self.setup_task()
            self.setup_env_context()
            self.setup_algo()
            self.setup_model()
            self.setup_replay_buffer()
            self.setup_online()
            
            # Load checkpoint
            if self.cfg.pretrained_model_path is not None:
                self.load_checkpoint(self.cfg.pretrained_model_path)
            
            # Setup multi-objective optimization flag
            self.is_moo: bool = self.task.is_moo
            
            # Register worker environment (critical for model forward pass)
            set_worker_env("trainer", self)
            set_worker_env("env", self.env)
            set_worker_env("ctx", self.ctx)
            set_worker_env("algo", self.algo)
            set_worker_env("task", self.task)
        else:
            super().setup()
    
    def setup_task(self):
        self.task = UniDockBoltzMOOTask(cfg=self.cfg)

    def build_training_data_loader(self):
        """Override to add BoltzinaSQLiteLogHook for separate database."""
        from torch.utils.data import DataLoader
        import pathlib
        
        # Get the parent's implementation logic
        model = self._wrap_for_mp(self.sampling_model)
        replay_buffer = self._wrap_for_mp(self.replay_buffer)

        if self.cfg.replay.use:
            assert self.cfg.replay.num_from_replay != 0, "Replay is enabled but no samples are being drawn from it"
            assert self.cfg.replay.num_new_samples != 0, "Replay is enabled but no new samples are being added to it"

        n_drawn = self.cfg.algo.num_from_policy
        n_replayed = self.cfg.replay.num_from_replay or n_drawn if self.cfg.replay.use else 0
        n_new_replay_samples = self.cfg.replay.num_new_samples or n_drawn if self.cfg.replay.use else None
        n_from_dataset = self.cfg.algo.num_from_dataset

        src = self.create_data_source(replay_buffer=replay_buffer)
        if n_from_dataset:
            src.do_sample_dataset(self.training_data, n_from_dataset, backwards_model=model)
        if n_drawn:
            src.do_sample_model(model, n_drawn, n_new_replay_samples)
        if n_replayed and replay_buffer is not None:
            src.do_sample_replay(n_replayed)
        if self.cfg.log_dir:
            # Add the standard SQLiteLogHook (from parent - uses CustomSQLiteLogHook)
            from rxnflow.base.gflownet.sqlite_log import CustomSQLiteLogHook
            src.add_sampling_hook(CustomSQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "train"), self.ctx))
            # Add the new BoltzinaSQLiteLogHook for separate database (works for boltz co-folding too)
            src.add_sampling_hook(BoltzinaSQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "train"), self.task))
        for hook in self.sampling_hooks:
            src.add_sampling_hook(hook)
        return self._make_data_loader(src)

    def log(self, info, index, key):
        self.add_extra_info(info)
        super().log(info, index, key)

    def add_extra_info(self, info):
        for prop, fr in self.task.avg_reward_info.items():
            info[f"sample_r_{prop}_avg"] = fr
        if len(self.task.batch_affinity) > 0:
            info["sample_boltz_avg"] = np.mean(self.task.batch_affinity)
        best_boltz = list(self.task.topn_affinity.values())
        for topn in [10, 100, 1000]:
            if len(best_boltz) > topn:
                info[f"top{topn}_boltz"] = np.mean(best_boltz[:topn])
