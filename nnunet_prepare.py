import os
import json
import shutil
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

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
    splits_json_path: Path = None
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
    
    # Get all patient cases from T2 directory
    t2_files = list(t2_dir.glob("*.nii.gz"))
    cases = [f.name.replace(".nii.gz", "") for f in t2_files]
    
    print(f"Found {len(cases)} cases in {t2_dir}")
    
    valid_cases = []
    
    for case in tqdm(cases, desc="Formatting cases"):
        t2_file = t2_dir / f"{case}.nii.gz"
        adc_file = adc_dir / f"{case}.nii.gz"
        hbv_file = hbv_dir / f"{case}.nii.gz"
        lesion_file = lesion_dir / f"{case}.nii.gz"
        
        # Check if core modalities exist
        if not all(p.exists() for p in [t2_file, adc_file, hbv_file]):
            print(f"Skipping {case}: Missing core modalities.")
            continue
            
        valid_cases.append(case)
        
        # nnUNet requires channels to be appended as _0000, _0001, etc.
        # Create Symlinks instead of copying to save disk space if possible, 
        # but shutil.copy is safer across different filesystems.
        # We will use symlinks to save DGX storage (if supported) or copy.
        
        dest_t2 = imagesTr / f"{case}_0000.nii.gz"
        dest_adc = imagesTr / f"{case}_0001.nii.gz"
        dest_hbv = imagesTr / f"{case}_0002.nii.gz"
        dest_label = labelsTr / f"{case}.nii.gz"
        
        try:
            if not dest_t2.exists(): os.symlink(t2_file, dest_t2)
            if not dest_adc.exists(): os.symlink(adc_file, dest_adc)
            if not dest_hbv.exists(): os.symlink(hbv_file, dest_hbv)
        except OSError:
            # Fallback to copy if symlink fails
            if not dest_t2.exists(): shutil.copy(t2_file, dest_t2)
            if not dest_adc.exists(): shutil.copy(adc_file, dest_adc)
            if not dest_hbv.exists(): shutil.copy(hbv_file, dest_hbv)

        # Handle Label
        if not dest_label.exists():
            if lesion_file.exists():
                try:
                    os.symlink(lesion_file, dest_label)
                except OSError:
                    shutil.copy(lesion_file, dest_label)
            else:
                # If lesion mask is missing (clinically negative), we must generate a zero mask for nnUNet
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
        shutil.copy(splits_json_path, dest_splits)
        print(f"Copied custom cross-validation splits to {dest_splits}")

if __name__ == '__main__':
    # Local DGX paths
    # These should be adjusted by the user to point to their DGX mounted paths
    SOURCE_DATA_DIR = Path("/Users/hemishjain/Desktop/PI-CAI/baseline/data")
    NNUNET_RAW_DIR = Path("/Users/hemishjain/Desktop/PI-CAI/baseline/nnUNet_raw_data_base/nnUNet_raw_data")
    SPLITS_FILE = Path("/Users/hemishjain/Desktop/PI-CAI/baseline/splits.json")
    
    prepare_nnunet_data(
        source_dir=SOURCE_DATA_DIR,
        nnunet_raw_dir=NNUNET_RAW_DIR,
        task_name="Task2302_z-nnmnet",
        splits_json_path=SPLITS_FILE
    )
