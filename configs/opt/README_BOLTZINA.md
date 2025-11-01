# How to Run UniDock-Boltzina Finetuning

## Prerequisites

1. **Activate the CGFlow conda environment:**
   ```bash
   conda activate cgflow-env
   ```

2. **Ensure the following conda environments exist:**
   - `cgflow-env` (active environment)
   - `unidock-env` (for UniDock docking)
   - `boltzina` (for Boltz-2 prediction and Boltzina scoring)

3. **Prepare Boltz-2 base.yaml config file:**
   - You need a Boltz-2 base.yaml configuration file
   - Update the path in the config file under `boltzina.base_yaml`

4. **Identify target residues:**
   - Determine the binding site residues from your protein structure
   - Format: `['A:123', 'A:124']` where `A` is chain ID and `123` is residue number
   - Update `boltzina.target_residues` in the config file

## Configuration File Setup

1. **Copy the example config:**
   ```bash
   cp cgflow/configs/opt/NS5_crop_boltzina.yaml cgflow/configs/opt/NS5_crop_boltzina_custom.yaml
   ```

2. **Edit the config file and update:**
   - `boltzina.base_yaml`: Path to your Boltz-2 base.yaml config file
   - `boltzina.target_residues`: List of actual binding site residues
   - Other parameters as needed

## Running the Training

```bash
cd cgflow
conda activate cgflow-env

python scripts/opt/opt_unidock_boltzina.py \
    --config configs/opt/NS5_crop_boltzina.yaml
```

## Or with command-line overrides:

```bash
python scripts/opt/opt_unidock_boltzina.py \
    --config configs/opt/NS5_crop_boltzina.yaml \
    --boltzina_target_residues A:123 A:124 A:125 \
    --boltzina_base_yaml /path/to/boltzina/configs/base.yaml
```

## What Happens During Training

1. **Pre-training Setup (automatic):**
   - Predicts protein structure using Boltz-2 in `boltzina` conda environment
   - Generates grid box from target residues
   - Optionally docks reference ligand for validation

2. **Training Loop:**
   - CGFlow generates molecules (in `cgflow-env`)
   - UniDock docks molecules (in `unidock-env`)
   - Boltzina scores docked structures (in `boltzina` conda environment)
   - Calculates Boltz score: `max(((-affinity_pred_value1+2)/4),0) * affinity_probability_binary1`
   - Multi-objective optimization: Boltz score × QED

## Output

Results will be saved to:
```
{result_dir}/{timestamp}/
├── boltzina_setup/          # Pre-training setup files
│   ├── predicted_receptor.pdb
│   ├── grid_config_*.txt
│   └── setup_info.json
├── boltzina_scoring/        # Scoring results during training
└── checkpoints/             # Model checkpoints
```

## Troubleshooting

1. **Conda environments not found:**
   - Ensure all three environments (`cgflow-env`, `unidock-env`, `boltzina`) exist
   - The script will automatically detect conda installation

2. **Boltz-2 prediction fails:**
   - Check that `boltzina.base_yaml` path is correct
   - Ensure `boltzina` conda environment has Boltz-2 installed

3. **Target residues not found:**
   - Verify residue numbers and chain IDs match your protein structure
   - Check PDB file format and chain assignments

4. **UniDock/Boltzina wrapper errors:**
   - Check that wrapper scripts exist in `cgflow/src/synthflow/utils/`
   - Ensure conda environments have required packages installed

