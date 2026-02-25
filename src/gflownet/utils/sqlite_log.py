import os
import sqlite3
from collections.abc import Iterable

import torch


class SQLiteLogHook:
    def __init__(self, log_dir, ctx) -> None:
        self.log = None  # Only initialized in __call__, which will occur inside the worker
        self.log_dir = log_dir
        self.ctx = ctx
        self.data_labels = None

    def __call__(self, trajs, rewards, obj_props, cond_info):
        if self.log is None:
            worker_info = torch.utils.data.get_worker_info()
            self._wid = worker_info.id if worker_info is not None else 0
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = f"{self.log_dir}/generated_objs_{self._wid}.db"
            self.log = SQLiteLog()
            self.log.connect(self.log_path)

        if hasattr(self.ctx, "object_to_log_repr"):
            objs = [self.ctx.object_to_log_repr(t["result"]) if t["is_valid"] else "" for t in trajs]
        else:
            objs = [""] * len(trajs)

        if hasattr(self.ctx, "traj_to_log_repr"):
            traj_str = [self.ctx.traj_to_log_repr(t["traj"]) if t["is_valid"] else "" for t in trajs]
        else:
            traj_str = [""] * len(trajs)

        obj_props = obj_props.reshape((len(obj_props), -1)).data.numpy().tolist()
        rewards = rewards.data.numpy().tolist()
        preferences = cond_info.get("preferences", torch.zeros((len(objs), 0))).data.numpy().tolist()
        focus_dir = cond_info.get("focus_dir", torch.zeros((len(objs), 0))).data.numpy().tolist()
        logged_keys = [k for k in sorted(cond_info.keys()) if k not in ["encoding", "preferences", "focus_dir"]]

        data = [
            [objs[i], rewards[i], traj_str[i]]
            + obj_props[i]
            + preferences[i]
            + focus_dir[i]
            + [cond_info[k][i].item() for k in logged_keys]
            for i in range(len(trajs))
        ]
        if self.data_labels is None:
            self.data_labels = (
                ["smi", "r", "traj"]
                + [f"fr_{i}" for i in range(len(obj_props[0]))]
                + [f"pref_{i}" for i in range(len(preferences[0]))]
                + [f"focus_{i}" for i in range(len(focus_dir[0]))]
                + [f"ci_{k}" for k in logged_keys]
            )

        self.log.insert_many(data, self.data_labels)
        return {}


class SQLiteLog:
    def __init__(self, timeout=300):
        """Creates a log instance, but does not connect it to any db."""
        self.is_connected = False
        self.db = None
        self.timeout = timeout

    def connect(self, db_path: str):
        """Connects to db_path

        Parameters
        ----------
        db_path: str
            The sqlite3 database path. If it does not exist, it will be created.
        """
        self.db = sqlite3.connect(db_path, timeout=self.timeout)
        cur = self.db.cursor()
        self._has_results_table = len(
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'").fetchall()
        )
        cur.close()

    def _make_results_table(self, types, names):
        type_map = {str: "text", float: "real", int: "real"}
        col_str = ", ".join(f"{name} {type_map[t]}" for t, name in zip(types, names, strict=False))
        cur = self.db.cursor()
        cur.execute(f"create table results ({col_str})")
        self._has_results_table = True
        cur.close()

    def insert_many(self, rows, column_names):
        assert all([isinstance(x, str) or not isinstance(x, Iterable) for x in rows[0]]), (
            "rows must only contain scalars"
        )
        if not self._has_results_table:
            self._make_results_table([type(i) for i in rows[0]], column_names)
        cur = self.db.cursor()
        cur.executemany(f"insert into results values ({','.join('?' * len(rows[0]))})", rows)  # nosec
        cur.close()
        self.db.commit()

    def __del__(self):
        if self.db is not None:
            self.db.close()


def read_all_results(path):
    # E402: module level import not at top of file, but pandas is an optional dependency
    import pandas as pd  # noqa: E402

    num_workers = len([f for f in os.listdir(path) if f.startswith("generated_objs")])
    dfs = [
        pd.read_sql_query("SELECT * FROM results", sqlite3.connect(f"file:{path}/generated_objs_{i}.db?mode=ro"))
        for i in range(num_workers)
    ]
    return pd.concat(dfs).sort_index().reset_index(drop=True)


class BoltzinaSQLiteLogHook:
    """
    Separate database hook for storing SMILES, docking scores, and all boltz scores.
    Creates a separate database file:
    - boltzina_scores_{worker_id}.db for UniDockBoltzinaTask
    - boltz_scores_{worker_id}.db for BoltzTask
    """
    def __init__(self, log_dir, task) -> None:
        self.log = None  # Only initialized in __call__, which will occur inside the worker
        self.log_dir = log_dir
        self.task = task
        self.data_labels = None
        
        # Determine database name based on task type
        task_class_name = task.__class__.__name__
        if "Boltzina" in task_class_name:
            # UniDockBoltzinaTask or UniDockBoltzinaMOOTask
            self.db_prefix = "boltzina_scores"
        elif "Boltz" in task_class_name:
            # BoltzTask or BoltzMOOTask
            self.db_prefix = "boltz_scores"
        else:
            # Default to boltzina for backward compatibility
            self.db_prefix = "boltzina_scores"

    def __call__(self, trajs, rewards, obj_props, cond_info):
        # Only log if task has batch_smiles (i.e., it's a UniDockBoltzinaTask or BoltzTask)
        if not hasattr(self.task, "batch_smiles") or not self.task.batch_smiles:
            return {}
        
        if self.log is None:
            worker_info = torch.utils.data.get_worker_info()
            self._wid = worker_info.id if worker_info is not None else 0
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = f"{self.log_dir}/{self.db_prefix}_{self._wid}.db"
            self.log = SQLiteLog()
            self.log.connect(self.log_path)

        # Get data from task
        smiles_list = self.task.batch_smiles
        docking_scores = self.task.batch_docking_scores
        boltz_scores = self.task.batch_boltz_scores
        
        # Get iteration from task or cond_info
        # Try to get from task first (set when batch data is computed)
        iteration = getattr(self.task, "batch_iteration", None)
        # If not in task, try to get from cond_info (if stored there)
        if iteration is None and "train_it" in cond_info:
            iteration = cond_info["train_it"][0].item() if len(cond_info["train_it"]) > 0 else None
        # Fallback: use 0 if not available
        if iteration is None:
            iteration = 0

        # Ensure all lists have the same length
        min_len = min(len(smiles_list), len(docking_scores), len(boltz_scores))
        if min_len == 0:
            return {}

        # Prepare data rows
        data = []
        for i in range(min_len):
            smiles = smiles_list[i] if i < len(smiles_list) else ""
            docking_score = docking_scores[i] if i < len(docking_scores) else 0.0
            
            # Extract boltz scores
            boltz_result = boltz_scores[i] if i < len(boltz_scores) else {}
            if isinstance(boltz_result, dict):
                affinity_ensemble = boltz_result.get("affinity_ensemble", 0.0)
                prob_ensemble = boltz_result.get("probability_ensemble", 0.0)
                affinity_model1 = boltz_result.get("affinity_model1", 0.0)
                prob_model1 = boltz_result.get("probability_model1", 0.0)
                affinity_model2 = boltz_result.get("affinity_model2", 0.0)
                prob_model2 = boltz_result.get("probability_model2", 0.0)
            else:
                # Fallback for old format
                affinity_ensemble = 0.0
                prob_ensemble = 0.0
                affinity_model1, prob_model1 = boltz_result if isinstance(boltz_result, tuple) else (0.0, 0.0)
                affinity_model2 = 0.0
                prob_model2 = 0.0

            data.append([
                int(iteration),  # iteration number
                smiles,
                float(docking_score),
                float(affinity_ensemble),
                float(prob_ensemble),
                float(affinity_model1),
                float(prob_model1),
                float(affinity_model2),
                float(prob_model2),
            ])

        if self.data_labels is None:
            self.data_labels = [
                "iteration",
                "smiles",
                "docking_score",
                "affinity_ensemble",
                "probability_ensemble",
                "affinity_model1",
                "probability_model1",
                "affinity_model2",
                "probability_model2",
            ]

        if data:
            self.log.insert_many(data, self.data_labels)
        
        return {}
