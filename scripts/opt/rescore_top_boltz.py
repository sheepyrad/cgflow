#!/usr/bin/env python3
"""
Re-score top N compounds from boltzina_scores_0.db using boltz (not boltzina).

This script:
1. Locates boltzina_scores_0.db in the specified directory (searches in train/ subdirectory)
2. Extracts top N compounds sorted by boltz_score (default: 100)
   - Boltz score formula: max(((-affinity_model1 + 2) / 4), 0) * probability_model1
3. Automatically finds receptor PDB, MSA file, and target residues from setup_info.json and config.yaml
4. Re-scores each compound using boltz predict with the same hotspot residues
5. Calculates boltz_score from re-scored results using the same formula
6. Saves results to a CSV file with both original boltzina scores and re-scored boltz scores

Usage:
    python rescore_top_boltz.py --dir /path/to/training/directory [OPTIONS]
    
Options:
    --db DB_FILE           Path to boltzina_scores database file (required)
    --top-n N              Number of top compounds to re-score (default: 100)
    --output OUTPUT.csv     Output CSV file path (default: <db_dir>/rescore_boltz_results.csv)
    --output-dir DIR        Directory for boltz output files (default: <db_dir>/rescore_boltz)

Note:
    This script is hardcoded for NS5 protein:
    - MSA: /home/conrad_hku/Drug_pipeline/msa/NS5_crop.a3m
    - Target residues: A:16, A:67, A:138, A:153, A:184, A:185
    - Protein sequence: QQVPFCSHHFHELIMKDGRKLVVPCRPQDELIGRARISQGAGWSLKETACLGKAYAQMWALMYFHRRDLRLASNAICSAVPVHWVPTSRTTWSIHAHHQWMTTEDMLTVWNRVWIEDNPWMEDKTPVTTWEDVPYLGKREDQWCGSLIGLTSRATWAQNILTAIQQVRSLIGNEEFLDYMPSMKRFR

Example:
    python rescore_top_boltz.py --db ./result/opt/unidock_boltzina/NS5/251121_165223/train/boltzina_scores_0.db
    python rescore_top_boltz.py --db ./result/opt/unidock_boltzina/NS5/251121_165223/train/boltzina_scores_0.db --top-n 50 --output-dir ./boltz_output
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import sqlite3
import yaml




def calculate_boltz_score(affinity: float, probability: float) -> float:
    """Calculate boltz_score from affinity and probability.
    
    Formula: max(((-affinity + 2) / 4), 0) * probability
    This matches the calculation in unidock_boltzina.py and streamlit dashboard.
    """
    normalized_aff = max(((-affinity + 2) / 4), 0.0)
    boltz_score = normalized_aff * probability
    return boltz_score


def load_top_compounds(db_path: Path, top_n: int = 100) -> pd.DataFrame:
    """Load top N compounds from database, sorted by boltz_score.
    
    Boltz score is calculated as: max(((-affinity_model1 + 2) / 4), 0) * probability_model1
    """
    conn = sqlite3.connect(str(db_path))
    
    # Check if table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'")
    if not cursor.fetchone():
        raise ValueError(f"No 'results' table found in {db_path}")
    
    # Get column names
    cursor.execute("PRAGMA table_info(results)")
    columns = [row[1] for row in cursor.fetchall()]
    
    # Check if required columns exist
    required_cols = ["affinity_model1", "probability_model1"]
    missing_cols = [col for col in required_cols if col not in columns]
    if missing_cols:
        raise ValueError(f"Required columns not found in database: {missing_cols}. Available columns: {columns}")
    
    # Load all data to calculate boltz_score
    query = "SELECT * FROM results"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return df
    
    # Calculate boltz_score for each compound
    df['boltz_score'] = df.apply(
        lambda row: calculate_boltz_score(
            row.get('affinity_model1', 0.0),
            row.get('probability_model1', 0.0)
        ),
        axis=1
    )
    
    # Sort by boltz_score descending (higher is better) and take top N
    df = df.sort_values('boltz_score', ascending=False).head(top_n)
    
    return df.reset_index(drop=True)




# Hardcoded NS5 values (matching boltz_screen_single.sh)
NS5_MSA_PATH = Path("/home/conrad_hku/Drug_pipeline/msa/NS5_crop.a3m")
NS5_PROTEIN_SEQUENCE = "QQVPFCSHHFHELIMKDGRKLVVPCRPQDELIGRARISQGAGWSLKETACLGKAYAQMWALMYFHRRDLRLASNAICSAVPVHWVPTSRTTWSIHAHHQWMTTEDMLTVWNRVWIEDNPWMEDKTPVTTWEDVPYLGKREDQWCGSLIGLTSRATWAQNILTAIQQVRSLIGNEEFLDYMPSMKRFR"
NS5_CONTACTS = [["A", 16], ["A", 67], ["A", 138], ["A", 153], ["A", 184], ["A", 185]]




def create_boltz_input_yaml(
    sequence: str,
    smiles: str,
    msa_path: Optional[Path],
    contacts: list[list],
    output_path: Path,
) -> None:
    """Create input.yaml file for boltz predict."""
    yaml_data = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": sequence,
                }
            },
            {
                "ligand": {
                    "id": "B",
                    "smiles": smiles,
                }
            }
        ],
        "properties": [
            {
                "affinity": {
                    "binder": "B"
                }
            }
        ],
        "constraints": [
            {
                "pocket": {
                    "binder": "B",
                    "contacts": contacts
                }
            }
        ]
    }
    
    # Add MSA if available
    if msa_path and msa_path.exists():
        yaml_data["sequences"][0]["protein"]["msa"] = str(msa_path.resolve())
    
    with open(output_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)


def run_boltz_predict(input_yaml: Path, output_dir: Path, query_name: Optional[str] = None) -> dict:
    """Run boltz predict and extract affinity score.
    
    Boltz-2 outputs affinity JSON files in:
    output_dir/predictions/{query_name}/affinity_{query_name}.json
    
    The JSON contains:
    - affinity_pred_value (ensemble)
    - affinity_probability_binary (ensemble)
    - affinity_pred_value1 (model1)
    - affinity_probability_binary1 (model1)
    - affinity_pred_value2 (model2)
    - affinity_probability_binary2 (model2)
    """
    try:
        print(f"    Running boltz predict (this may take a while)...")
        print(f"    Input YAML: {input_yaml}")
        print(f"    Output directory: {output_dir}")
        print(f"    Query name: {query_name}")
        
        result = subprocess.run(
            [
                "boltz",
                "predict",
                str(input_yaml),
                "--out_dir",
                str(output_dir),
                "--output_format",
                "pdb",
                "--use_potentials",
                "--affinity_mw_correction",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=None,  # No timeout - let it run as long as needed
        )
        
        print(f"    Boltz predict completed (exit code: {result.returncode})")
        
        # Print first few lines of stdout/stderr for debugging
        if result.stdout:
            stdout_lines = result.stdout.strip().split("\n")
            if stdout_lines:
                print(f"    Boltz stdout (last 5 lines):")
                for line in stdout_lines[-5:]:
                    if line.strip():
                        print(f"      {line}")
        
        if result.stderr:
            stderr_lines = result.stderr.strip().split("\n")
            if stderr_lines:
                print(f"    Boltz stderr (last 5 lines):")
                for line in stderr_lines[-5:]:
                    if line.strip():
                        print(f"      {line}")
        
        # Parse output to extract affinity
        # Boltz-2 outputs affinity in JSON format
        # Try to find affinity JSON file
        affinity_file = None
        
        # Try multiple possible output structures (similar to UniDockBoltzTask._extract_boltz_results)
        possible_paths = []
        
        # Structure 1: <out_dir>/predictions/<query_name>/affinity_<query_name>.json
        if query_name:
            possible_paths.append(output_dir / "predictions" / query_name / f"affinity_{query_name}.json")
            # Structure 2: <out_dir>/predictions/<query_name>/affinity_*.json (any affinity file)
            predictions_dir = output_dir / "predictions" / query_name
            if predictions_dir.exists():
                possible_paths.extend(predictions_dir.glob("affinity_*.json"))
        
        # Structure 3: <out_dir>/boltz_results_*/predictions/<query_name>/affinity_*.json
        if query_name:
            boltz_results_dirs = list(output_dir.glob("boltz_results_*/predictions"))
            for br_dir in boltz_results_dirs:
                query_path = br_dir / query_name
                if query_path.exists():
                    possible_paths.extend(query_path.glob("affinity_*.json"))
        
        # Structure 4: <out_dir>/boltz_results_*/predictions/*/affinity_*.json (any query name)
        boltz_results_dirs = list(output_dir.glob("boltz_results_*/predictions"))
        for br_dir in boltz_results_dirs:
            if br_dir.exists():
                # Look for any subdirectories with affinity files
                for subdir in br_dir.iterdir():
                    if subdir.is_dir():
                        possible_paths.extend(subdir.glob("affinity_*.json"))
        
        # Structure 5: Search recursively for any affinity JSON files (but exclude processed/manifest.json)
        if not possible_paths:
            all_affinity_files = list(output_dir.rglob("affinity_*.json"))
            # Filter out manifest.json files
            possible_paths.extend([f for f in all_affinity_files if "manifest" not in str(f)])
        
        # Find the first existing affinity file
        for path in possible_paths:
            if path.exists():
                affinity_file = path
                break
        
        if affinity_file and affinity_file.exists():
            try:
                with open(affinity_file, "r") as f:
                    affinity_data = json.load(f)
                
                # Extract all affinity scores
                result_dict = {
                    "affinity_ensemble": float(affinity_data.get("affinity_pred_value", 0.0)),
                    "probability_ensemble": float(affinity_data.get("affinity_probability_binary", 0.0)),
                    "affinity_model1": float(affinity_data.get("affinity_pred_value1", 0.0)),
                    "probability_model1": float(affinity_data.get("affinity_probability_binary1", 0.0)),
                    "affinity_model2": float(affinity_data.get("affinity_pred_value2", 0.0)),
                    "probability_model2": float(affinity_data.get("affinity_probability_binary2", 0.0)),
                    "success": True,
                }
                # Use ensemble affinity as primary score
                result_dict["affinity"] = result_dict["affinity_ensemble"]
                return result_dict
            except Exception as e:
                return {
                    "affinity": None,
                    "success": False,
                    "error": f"Failed to parse affinity JSON: {e}",
                }
        
        # If no JSON found, try to parse from stdout
        # Boltz-2 might print affinity in stdout
        if "affinity" in result.stdout.lower() or "predicted" in result.stdout.lower():
            # Try to extract number from output
            import re
            numbers = re.findall(r"-?\d+\.?\d*", result.stdout)
            if numbers:
                # Usually affinity is negative, so take the most negative reasonable value
                affinities = [float(n) for n in numbers if abs(float(n)) < 100]
                if affinities:
                    return {"affinity": min(affinities), "success": True, "note": "Parsed from stdout"}
        
        # If we can't parse, print debug info and return failure
        print(f"    ERROR: Could not find affinity JSON file in {output_dir}")
        print(f"    Searched for query_name: {query_name}")
        print(f"    Output directory exists: {output_dir.exists()}")
        if output_dir.exists():
            print(f"    Output directory contents:")
            items = list(output_dir.iterdir())
            for item in items[:20]:  # Show first 20 items
                item_type = "dir" if item.is_dir() else "file"
                size = item.stat().st_size if item.is_file() else 0
                print(f"      {item.name} ({item_type}, {size} bytes)")
            if len(items) > 20:
                print(f"      ... and {len(items) - 20} more items")
            
            # Also check recursively for any JSON files
            json_files = list(output_dir.rglob("*.json"))
            if json_files:
                print(f"    Found {len(json_files)} JSON files:")
                for json_file in json_files[:10]:
                    print(f"      {json_file.relative_to(output_dir)}")
        else:
            print(f"      Output directory does not exist!")
        
        return {
            "affinity": None,
            "success": False,
            "error": f"Could not find affinity JSON file in {output_dir}. Check boltz output above.",
        }
        
    except subprocess.CalledProcessError as e:
        print(f"    Error: boltz predict failed with exit code {e.returncode}")
        if e.stdout:
            print(f"    Stdout: {e.stdout[-500:]}")  # Last 500 chars
        if e.stderr:
            print(f"    Stderr: {e.stderr[-500:]}")  # Last 500 chars
        return {
            "affinity": None,
            "success": False,
            "error": e.stderr or str(e),
        }
    except FileNotFoundError:
        return {
            "affinity": None,
            "success": False,
            "error": "boltz command not found. Please activate the Boltz environment.",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Re-score top compounds from boltzina_scores database using boltz"
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to boltzina_scores database file (e.g., train/boltzina_scores_0.db)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top compounds to re-score (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: <db_dir>/rescore_boltz_results.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for boltz output files (default: <db_dir>/rescore_boltz)",
    )
    
    args = parser.parse_args()
    
    # Validate database file
    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Using database: {db_path}")
    
    # Load top compounds
    print(f"Loading top {args.top_n} compounds...")
    try:
        df = load_top_compounds(db_path, args.top_n)
        print(f"Loaded {len(df)} compounds")
    except Exception as e:
        print(f"Error loading compounds: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Use hardcoded NS5 values
    msa_path = NS5_MSA_PATH
    if not msa_path.exists():
        print(f"Error: MSA file not found: {msa_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Using MSA file: {msa_path}")
    
    # Use hardcoded NS5 protein sequence
    sequence = NS5_PROTEIN_SEQUENCE
    print(f"Using NS5 protein sequence (length: {len(sequence)})")
    
    # Use hardcoded NS5 contacts
    contacts = NS5_CONTACTS
    print(f"Using NS5 contacts: {contacts}")
    
    # Determine output directories
    db_dir = db_path.parent
    if args.output_dir:
        output_base_dir = Path(args.output_dir).resolve()
    else:
        output_base_dir = db_dir / "rescore_boltz"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Boltz output directory: {output_base_dir}")
    
    # Re-score compounds
    print(f"\nRe-scoring {len(df)} compounds with boltz...")
    results = []
    
    for idx, row in df.iterrows():
        smiles = row["smiles"]
        compound_id = row.get("rowid", idx)
        
        # Generate unique ID for this compound
        smiles_hash = hashlib.md5(smiles.encode()).hexdigest()[:8]
        lig_id = f"compound_{compound_id}_{smiles_hash}"
        
        print(f"[{idx+1}/{len(df)}] Processing {lig_id}: {smiles[:50]}...")
        
        # Create output directory for this compound
        compound_output_dir = output_base_dir / lig_id
        compound_output_dir.mkdir(exist_ok=True)
        
        # Create input.yaml
        input_yaml = compound_output_dir / "input.yaml"
        try:
            create_boltz_input_yaml(
                sequence=sequence,
                smiles=smiles,
                msa_path=msa_path,
                contacts=contacts,
                output_path=input_yaml,
            )
        except Exception as e:
            print(f"  Error creating input.yaml: {e}", file=sys.stderr)
            # Get original boltz_score if available
            original_boltz_score = row.get("boltz_score")
            if original_boltz_score is None and row.get("affinity_model1") is not None:
                original_boltz_score = calculate_boltz_score(
                    row.get("affinity_model1", 0.0),
                    row.get("probability_model1", 0.0)
                )
            results.append({
                "compound_id": compound_id,
                "smiles": smiles,
                "original_boltz_score": original_boltz_score,
                "original_affinity_model1": row.get("affinity_model1"),
                "original_probability_model1": row.get("probability_model1"),
                "original_affinity_ensemble": row.get("affinity_ensemble"),
                "original_probability_ensemble": row.get("probability_ensemble"),
                "original_docking_score": row.get("docking_score"),
                "rescore_boltz_score": None,
                "rescore_affinity_model1": None,
                "rescore_probability_model1": None,
                "success": False,
                "error": f"Failed to create input.yaml: {e}",
            })
            continue
        
        # Run boltz predict
        # Query name is typically derived from input YAML filename or ligand ID
        # Boltz-2 uses the base name of the input YAML file
        query_name = input_yaml.stem  # Remove .yaml extension
        boltz_result = run_boltz_predict(input_yaml, compound_output_dir, query_name=query_name)
        
        # Calculate boltz_score from re-scored results
        rescore_boltz_score = None
        if boltz_result.get("success") and boltz_result.get("affinity_model1") is not None:
            rescore_boltz_score = calculate_boltz_score(
                boltz_result.get("affinity_model1", 0.0),
                boltz_result.get("probability_model1", 0.0)
            )
        
        # Get original boltz_score if available, otherwise calculate it
        original_boltz_score = row.get("boltz_score")
        if original_boltz_score is None and row.get("affinity_model1") is not None:
            original_boltz_score = calculate_boltz_score(
                row.get("affinity_model1", 0.0),
                row.get("probability_model1", 0.0)
            )
        
        # Store result
        result_row = {
            "compound_id": compound_id,
            "smiles": smiles,
            "original_boltz_score": original_boltz_score,
            "original_affinity_model1": row.get("affinity_model1"),
            "original_probability_model1": row.get("probability_model1"),
            "original_affinity_ensemble": row.get("affinity_ensemble"),
            "original_probability_ensemble": row.get("probability_ensemble"),
            "original_docking_score": row.get("docking_score"),
            "rescore_boltz_score": rescore_boltz_score,
            "rescore_affinity_model1": boltz_result.get("affinity_model1"),
            "rescore_probability_model1": boltz_result.get("probability_model1"),
            "rescore_affinity_model2": boltz_result.get("affinity_model2"),
            "rescore_probability_model2": boltz_result.get("probability_model2"),
            "rescore_affinity_ensemble": boltz_result.get("affinity_ensemble"),
            "rescore_probability_ensemble": boltz_result.get("probability_ensemble"),
            "success": boltz_result.get("success", False),
        }
        
        if "error" in boltz_result:
            result_row["error"] = boltz_result["error"]
        if "note" in boltz_result:
            result_row["note"] = boltz_result["note"]
        
        results.append(result_row)
        
        if boltz_result.get("success"):
            affinity = boltz_result.get("affinity_model1")
            prob = boltz_result.get("probability_model1")
            boltz_score = rescore_boltz_score
            # Format values safely, handling None
            affinity_str = f"{affinity:.4f}" if affinity is not None else "N/A"
            prob_str = f"{prob:.4f}" if prob is not None else "N/A"
            boltz_score_str = f"{boltz_score:.4f}" if boltz_score is not None else "N/A"
            print(f"  ✓ Affinity: {affinity_str}, Probability: {prob_str}, Boltz Score: {boltz_score_str}")
        else:
            print(f"  ✗ Failed: {boltz_result.get('error', 'Unknown error')}")
    
    # Save results to CSV
    output_csv = args.output
    if not output_csv:
        output_csv = db_dir / "rescore_boltz_results.csv"
    else:
        output_csv = Path(output_csv)
    
    print(f"\nSaving results to {output_csv}...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    
    # Print summary
    successful = sum(1 for r in results if r.get("success", False))
    successful_results = [r for r in results if r.get("success", False) and r.get("rescore_boltz_score") is not None]
    
    print(f"\nSummary:")
    print(f"  Total compounds: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(results) - successful}")
    
    if successful_results:
        rescore_scores = [r["rescore_boltz_score"] for r in successful_results]
        original_scores = [r.get("original_boltz_score") for r in successful_results if r.get("original_boltz_score") is not None]
        
        print(f"\nBoltz Score Statistics (re-scored):")
        print(f"  Mean: {sum(rescore_scores) / len(rescore_scores):.4f}")
        print(f"  Max: {max(rescore_scores):.4f}")
        print(f"  Min: {min(rescore_scores):.4f}")
        
        if original_scores:
            print(f"\nOriginal vs Re-scored Comparison:")
            print(f"  Original mean: {sum(original_scores) / len(original_scores):.4f}")
            print(f"  Re-scored mean: {sum(rescore_scores) / len(rescore_scores):.4f}")
            print(f"  Difference: {(sum(rescore_scores) / len(rescore_scores)) - (sum(original_scores) / len(original_scores)):.4f}")
    
    print(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    main()

