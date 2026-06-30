import csv
import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from collections import defaultdict

def create_splits(marksheet_path: Path, output_splits_path: Path, train_centers: list, n_splits: int = 5):
    print(f"Generating domain adaptation splits for centers: {train_centers}")
    
    # Read marksheet
    patient_to_cases = defaultdict(list)
    case_centers = {}
    
    try:
        with open(marksheet_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                center = row.get("center")
                patient_id = row.get("patient_id")
                study_id = row.get("study_id")
                
                if not center or not patient_id or not study_id:
                    continue
                    
                case_id = f"{patient_id}_{study_id}"
                case_centers[case_id] = center
                
                # Only include cases from the requested train centers
                if center in train_centers:
                    patient_to_cases[patient_id].append(case_id)
    except FileNotFoundError:
        print(f"ERROR: Could not find {marksheet_path}")
        return False
        
    patients = sorted(list(patient_to_cases.keys()))
    print(f"Found {len(patients)} unique patients from selected centers.")
    
    if len(patients) < n_splits:
        print(f"ERROR: Not enough patients ({len(patients)}) to create {n_splits} folds.")
        return False
        
    # GroupKFold logic using standard KFold on unique patients
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    splits = []
    for train_idx, val_idx in kf.split(patients):
        train_patients = [patients[i] for i in train_idx]
        val_patients = [patients[i] for i in val_idx]
        
        train_cases = []
        for p in train_patients:
            train_cases.extend(patient_to_cases[p])
            
        val_cases = []
        for p in val_patients:
            val_cases.extend(patient_to_cases[p])
            
        splits.append({
            "train": sorted(train_cases),
            "val": sorted(val_cases)
        })
        
    with open(output_splits_path, 'w') as f:
        json.dump(splits, f, indent=4)
        
    print(f"Successfully wrote 5-fold cross-validation splits to {output_splits_path}")
    for i, fold in enumerate(splits):
        print(f"  Fold {i}: {len(fold['train'])} train cases, {len(fold['val'])} val cases")
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create domain adaptation splits for PI-CAI.")
    parser.add_argument("--marksheet", type=str, required=True, help="Path to marksheet.csv")
    parser.add_argument("--output", type=str, required=True, help="Path to output splits.json")
    parser.add_argument("--centers", type=str, required=True, help="Comma-separated list of centers to train on (e.g. 'RUMC,ZGT')")
    args = parser.parse_args()
    
    centers_list = [c.strip() for c in args.centers.split(",")]
    
    create_splits(
        marksheet_path=Path(args.marksheet),
        output_splits_path=Path(args.output),
        train_centers=centers_list
    )
