from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED

from gflownet.utils import sascore

from synthflow.config import Config
from synthflow.pocket_specific.trainer import RxnFlow3DTrainer_single
from synthflow.utils import boltzina_integration, unidock

from .docking import BaseDockingTask


class UniDockBoltzinaTask(BaseDockingTask):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # Get Boltzina-specific config
        self.boltzina_receptor_pdb = Path(cfg.task.boltzina.receptor_pdb)
        self.boltzina_work_dir = Path(cfg.task.boltzina.work_dir)
        self.boltzina_output_dir = Path(cfg.log_dir) / "boltzina_scoring"
        self.boltzina_output_dir.mkdir(parents=True, exist_ok=True)
        # Use getattr with defaults for optional fields
        self.boltzina_fname = getattr(cfg.task.boltzina, "fname", "cgflow_ligand")
        self.boltzina_batch_size = getattr(cfg.task.boltzina, "batch_size", 1)
        self.boltzina_num_workers = getattr(cfg.task.boltzina, "num_workers", 1)

    def calc_affinities(self, mols: list[Chem.Mol]) -> list[float]:
        if self.redocking:
            return self.run_redocking_with_boltzina(mols)
        else:
            raise NotImplementedError("Local optimization not implemented for UniDockBoltzinaTask")

    def run_redocking_with_boltzina(self, mols: list[Chem.Mol]) -> list[float]:
        """Run UniDock docking followed by Boltzina scoring."""
        # Step 1: Dock using UniDock
        try:
            res = unidock.docking(mols, self.protein_path, self.center, search_mode="balance")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f"Unidock is not installed. Please install it using conda. {e}") from e
        except Exception:
            return [0.0] * len(mols)

        # Save docked structures
        output_result_path = self.save_dir / f"oracle{self.oracle_idx}_dock.sdf"
        docked_mols = []
        with Chem.SDWriter(str(output_result_path)) as w:
            for docked_mol, _ in res:
                if docked_mol is None:
                    docked_mols.append(None)
                else:
                    docked_mols.append(docked_mol)
                    w.write(docked_mol)

        # Step 2: Score using Boltzina
        try:
            affinity_results = boltzina_integration.boltzina_scoring(
                docked_mols=docked_mols,
                receptor_pdb=self.boltzina_receptor_pdb,
                work_dir=self.boltzina_work_dir,
                output_dir=self.boltzina_output_dir / f"oracle{self.oracle_idx}",
                fname=self.boltzina_fname,
                batch_size=self.boltzina_batch_size,
                num_workers=self.boltzina_num_workers,
                seed=self.cfg.seed if hasattr(self.cfg, "seed") else None,
            )

            # Step 3: Calculate Boltz score
            boltz_scores = []
            for affinity_value1, affinity_prob1 in affinity_results:
                # Boltz score formula: max(((-affinity_pred_value1+2)/4),0) * affinity_probability_binary1
                normalized_aff = max(((-affinity_value1 + 2) / 4), 0.0)
                boltz_score = normalized_aff * affinity_prob1
                boltz_scores.append(boltz_score)

            return boltz_scores

        except Exception as e:
            print(f"Error in Boltzina scoring: {e}")
            return [0.0] * len(mols)


class UniDockBoltzinaMOOTask(UniDockBoltzinaTask):
    avg_reward_info: OrderedDict[str, float]

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.objectives = self.cfg.task.moo.objectives
        assert set(self.objectives) <= {"boltz", "qed", "sa"}
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


class UniDockBoltzinaMOOTrainer(RxnFlow3DTrainer_single[UniDockBoltzinaMOOTask]):
    def setup_task(self):
        self.task = UniDockBoltzinaMOOTask(cfg=self.cfg)

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

