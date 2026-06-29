import numpy as np
import SimpleITK as sitk
from pathlib import Path
import pandas as pd

# Base directories
BASE_DIR = Path("/Users/hemishjain/Desktop/PI-CAI/baseline/data")
CASE_ID = "10008_1000008"

def create_dummy_image(shape, spacing, is_mask=False, mask_val=1):
    arr = np.zeros(shape, dtype=np.float32)
    z_center, y_center, x_center = shape[0]//2, shape[1]//2, shape[2]//2
    
    if is_mask:
        # Create a block in the center
        arr[z_center-2:z_center+3, y_center-40:y_center+40, x_center-40:x_center+40] = mask_val
        arr = arr.astype(np.uint8)
    else:
        # Random background noise
        arr = np.random.normal(100, 20, shape).astype(np.float32)
        # Higher intensity block in the center
        arr[z_center-2:z_center+3, y_center-40:y_center+40, x_center-40:x_center+40] = 500

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img

def create_zonal_mask(shape, spacing):
    arr = np.zeros(shape, dtype=np.uint8)
    z_center, y_center, x_center = shape[0]//2, shape[1]//2, shape[2]//2
    
    # PZ = 1
    arr[z_center-2:z_center+3, y_center-40:y_center+40, x_center-40:x_center+40] = 1
    # TZ = 2 (smaller block inside PZ)
    arr[z_center-1:z_center+2, y_center-20:y_center+20, x_center-20:x_center+20] = 2

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img

print("Creating dummy NIfTI files...")
shape = (22, 384, 384) # z, y, x
spacing = (0.34, 0.34, 3.0) # original spacing

t2 = create_dummy_image(shape, spacing)
adc = create_dummy_image(shape, spacing)
hbv = create_dummy_image(shape, spacing)
gland = create_dummy_image(shape, spacing, is_mask=True, mask_val=1)
lesion = create_dummy_image(shape, spacing, is_mask=True, mask_val=1)
zonal = create_zonal_mask(shape, spacing)

sitk.WriteImage(t2, str(BASE_DIR / f"t2/{CASE_ID}.nii.gz"))
sitk.WriteImage(adc, str(BASE_DIR / f"adc_reg/{CASE_ID}.nii.gz"))
sitk.WriteImage(hbv, str(BASE_DIR / f"hbv_reg/{CASE_ID}.nii.gz"))
sitk.WriteImage(gland, str(BASE_DIR / f"whole_gland_masks/{CASE_ID}.nii.gz"))
sitk.WriteImage(lesion, str(BASE_DIR / f"lesion_masks/{CASE_ID}.nii.gz"))
sitk.WriteImage(zonal, str(BASE_DIR / f"zonal_masks/{CASE_ID}.nii.gz"))

print("Creating dummy marksheet...")
df = pd.DataFrame({
    'patient_id': [10000],
    'study_id': [1000000],
    'center': ['RUMC'],
    'age': [65],
    'PSA': [5.2],
    'prostate_volume': [40],
    'PSA_density': [0.13],
    'case_ISUP': [2],
    'case_csPCa': [1],
    'histopath_type': ['MR-targeted biopsy'],
    'days_to_histopathology': [14]
})
df.to_csv(BASE_DIR / "clinical_information/marksheet.csv", index=False)

print("Dummy data creation complete.")
