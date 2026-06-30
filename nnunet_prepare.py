import os
import json
import shutil
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import gzip

def copy_and_compress(src: Path, dest: Path):
    """Copies a file to dest. If the source is a raw .nii, it compresses it to .nii.gz on the fly."""
    if not src.exists():
        return
    if src.suffix == '.nii' and dest.name.endswith('.nii.gz'):
        with open(src, 'rb') as f_in:
            with gzip.open(dest, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copy2(src, dest)

def generate_empty_mask(reference_image_path: Path, output_path: Path):
    """Generates an empty (all zeros) mask matching the geometry of the reference image."""
    ref_img = sitk.ReadImage(str(reference_image_path))
    empty_arr = np.zeros(ref_img.GetSize()[::-1], dtype=np.uint8)
    empty_img = sitk.GetImageFromArray(empty_arr)
    empty_img.CopyInformation(ref_img)
    sitk.WriteImage(empty_img, str(output_path))

def prepare_nnunet_data(
    source_dir: Path, 
    nnunet_raw_dir: Path, 
    task_name: str = "Task2302_z-nnmnet",
    splits_json_path: Path = None,
    max_cases: int = None
):
    print(f"Preparing nnUNet dataset: {task_name}")
    
    task_dir = nnunet_raw_dir / task_name
    imagesTr = task_dir / "imagesTr"
    labelsTr = task_dir / "labelsTr"
    
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)
    
    t2_dir = source_dir / "t2"
    adc_dir = source_dir / "adc_reg"
    hbv_dir = source_dir / "hbv_reg"
    lesion_dir = source_dir / "lesion_masks"
    
    # Get all patient cases from T2 directory (support both .nii.gz and .nii)
    t2_files = list(t2_dir.glob("*.nii.gz"))
    if len(t2_files) == 0:
        t2_files = list(t2_dir.glob("*.nii"))
        
    cases = sorted([f.name.replace(".nii.gz", "").replace(".nii", "") for f in t2_files])
    
    print(f"Found {len(cases)} cases in {t2_dir}")
    
    # Optionally limit the number of cases for sanity testing
    if max_cases is not None and max_cases < len(cases):
        import random
        random.seed(42)
        cases = random.sample(cases, max_cases)
        print(f"Subset mode: randomly selected {max_cases} cases for processing.")
    
    valid_cases = []
    
    for case in tqdm(cases, desc="Formatting cases"):
        # Resolve exact source paths (they could be .nii or .nii.gz)
        t2_file = t2_dir / f"{case}.nii.gz"
        if not t2_file.exists(): t2_file = t2_dir / f"{case}.nii"
        
        adc_file = adc_dir / f"{case}.nii.gz"
        if not adc_file.exists(): adc_file = adc_dir / f"{case}.nii"
        
        hbv_file = hbv_dir / f"{case}.nii.gz"
        if not hbv_file.exists(): hbv_file = hbv_dir / f"{case}.nii"
        
        lesion_file = lesion_dir / f"{case}.nii.gz"
        if not lesion_file.exists(): lesion_file = lesion_dir / f"{case}.nii"
        
        # Check if core modalities exist
        if not all(p.exists() for p in [t2_file, adc_file, hbv_file]):
            print(f"Skipping {case}: Missing core modalities.")
            continue
            
        valid_cases.append(case)
        
        # nnUNet requires channels to be appended as _0000, _0001, etc.
        dest_t2 = imagesTr / f"{case}_0000.nii.gz"
        dest_adc = imagesTr / f"{case}_0001.nii.gz"
        dest_hbv = imagesTr / f"{case}_0002.nii.gz"
        dest_label = labelsTr / f"{case}.nii.gz"
        
        # Explicitly copy and compress on the fly if needed
        if not dest_t2.exists(): copy_and_compress(t2_file, dest_t2)
        if not dest_adc.exists(): copy_and_compress(adc_file, dest_adc)
        if not dest_hbv.exists(): copy_and_compress(hbv_file, dest_hbv)

        # Handle Label (and protect against corrupted PI-CAI mask files)
        if not dest_label.exists():
            is_valid_mask = False
            if lesion_file.exists() and lesion_file.stat().st_size > 348:
                try:
                    reader = sitk.ImageFileReader()
                    reader.SetFileName(str(lesion_file))
                    reader.ReadImageInformation()
                    is_valid_mask = True
                except Exception:
                    pass
            
            if is_valid_mask:
                copy_and_compress(lesion_file, dest_label)
            else:
                # If lesion mask is missing (clinically negative) or corrupted, generate a zero mask
                generate_empty_mask(t2_file, dest_label)
                
    # Generate dataset.json
    dataset_json = {
        "name": "PI-CAI Z-SSMNet",
        "description": "PI-CAI dataset formatted for Z-SSMNet nnUNet backbone",
        "reference": "https://github.com/yuanyuan29/Z-SSMNet",
        "licence": "Apache 2.0",
        "release": "1.0 29/06/2026",
        "tensorImageSize": "3D",
        "modality": {
            "0": "T2",
            "1": "ADC",
            "2": "HBV"
        },
        "labels": {
            "0": "background",
            "1": "csPCa"
        },
        "numTraining": len(valid_cases),
        "numTest": 0,
        "training": [
            {"image": f"./imagesTr/{case}.nii.gz", "label": f"./labelsTr/{case}.nii.gz"} for case in valid_cases
        ],
        "test": []
    }
    
    with open(task_dir / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=4)
        
    print(f"Successfully formatted {len(valid_cases)} cases for nnUNet.")
    
    # Copy splits.json if provided
    if splits_json_path and splits_json_path.exists():
        dest_splits = task_dir / "splits.json"
        shutil.copy2(splits_json_path, dest_splits)
        print(f"Copied custom cross-validation splits to {dest_splits}")

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare PI-CAI data for nnUNet.")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to raw preprocessed data in Google Drive.")
    parser.add_argument("--nnunet_raw_dir", type=str, required=True, help="Path to local NVMe workspace nnUNet_raw_data directory.")
    parser.add_argument("--splits_json", type=str, default=None, help="Path to custom splits.json.")
    parser.add_argument("--max_cases", type=int, default=None, help="Limit the number of cases to process (for sanity testing).")
    args = parser.parse_args()
    
    prepare_nnunet_data(
        source_dir=Path(args.source_dir),
        nnunet_raw_dir=Path(args.nnunet_raw_dir),
        task_name="Task2302_z-nnmnet",
        splits_json_path=Path(args.splits_json) if args.splits_json else None,
        max_cases=args.max_cases
    )
