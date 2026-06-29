import os
import pickle
import stat
from pathlib import Path
from typing import Union
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

def prepare_zonal_mask_npz(
    zonal_masks_dir: Path,
    nnunet_preprocessed_dir: Path,
):
    print(f"Injecting zonal masks from {zonal_masks_dir} into nnUNet cache at {nnunet_preprocessed_dir}")
    
    if not nnunet_preprocessed_dir.exists():
        print(f"Error: nnUNet preprocessed directory {nnunet_preprocessed_dir} does not exist. Did you run nnUNet_plan_and_preprocess?")
        return

    try:
        os.chmod(nnunet_preprocessed_dir, stat.S_IRWXO | stat.S_IRWXU)
    except PermissionError as e:
        print(e)
        
    pkl_files = list(nnunet_preprocessed_dir.glob("*.pkl"))
    print(f"Found {len(pkl_files)} preprocessed cases in cache.")

    for file in tqdm(pkl_files, desc="Processing Zonal Masks"):
        file_name = file.name[:13] # e.g. 10000_1000000
        zonal_mask_path = zonal_masks_dir / f"{file_name}.nii.gz"
        
        if not zonal_mask_path.exists():
            print(f"Warning: Zonal mask {zonal_mask_path} not found. Skipping...")
            continue
            
        with open(file, 'rb') as f:
            data = pickle.load(f)
            org_direction = data['itk_direction']
            crop_bbox = data['crop_bbox']
            crop_size = data['size_after_cropping']
            resample_spacing = data['spacing_after_resampling']
            resample_size = data['size_after_resampling']

            mask = sitk.ReadImage(str(zonal_mask_path))

            # Crop the mask to match nnUNet's dynamic crop
            roi_index = (crop_bbox[2][0], crop_bbox[1][0], crop_bbox[0][0])
            roi_size = (crop_size[2], crop_size[1], crop_size[0])
            
            try:
                mask_crop = sitk.RegionOfInterest(mask, roi_size, roi_index)
            except Exception as e:
                print(f"Error cropping {file_name}: {e}")
                continue

            # Resample the mask to match nnUNet's target spacing
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(mask_crop)
            resampler.SetOutputSpacing((resample_spacing[2], resample_spacing[1], resample_spacing[0]))
            resampler.SetOutputDirection(org_direction)
            resampler.SetSize((resample_size[2], resample_size[1], resample_size[0]))
            # CRITICAL: Use NearestNeighbor so we don't interpolate integers (0,1,2) into floats!
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            mask_resample = resampler.Execute(mask_crop)

            # Save the mask as _seg.npz
            mask_npz = sitk.GetArrayFromImage(mask_resample)
            
            # The custom myTrainer_zonal expects this file to exist alongside the image .npz
            np.savez_compressed(nnunet_preprocessed_dir / f"{file_name}_seg.npz", data=mask_npz)

if __name__ == "__main__":
    # Local DGX paths (User should adjust these)
    ZONAL_MASKS_DIR = Path("/Users/hemishjain/Desktop/PI-CAI/baseline/data/zonal_masks")
    
    # nnUNet typically saves preprocessed data here for 3D full resolution:
    NNUNET_PREPROCESSED_DIR = Path("/Users/hemishjain/Desktop/PI-CAI/baseline/nnUNet_preprocessed/Task2302_z-nnmnet/nnUNetData_plans_v2.1_stage0")
    
    prepare_zonal_mask_npz(
        zonal_masks_dir=ZONAL_MASKS_DIR,
        nnunet_preprocessed_dir=NNUNET_PREPROCESSED_DIR
    )
