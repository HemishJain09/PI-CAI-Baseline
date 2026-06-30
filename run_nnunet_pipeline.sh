#!/bin/bash
set -e

# ==============================================================================
# PI-CAI Z-SSMNet nnUNet Pipeline Orchestrator
# ==============================================================================

# 1. Define Paths 
# ------------------------------------------------------------------------------
# These can be overridden by environment variables from the Colab Notebook
WORKSPACE_DIR="${WORKSPACE_DIR:-/content/PI-CAI_Workspace/baseline}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-/content/drive/MyDrive/PI-CAI_pre-processed}"
CUSTOM_SPLITS_FILE="$WORKSPACE_DIR/splits.json"
TASK_NAME="Task2302_z-nnmnet"
TASK_ID="2302"

# Subset testing: set MAX_CASES to limit number of patients (e.g. 10 for sanity check)
MAX_CASES="${MAX_CASES:-}"
# Training epochs: set MAX_EPOCHS to override the default 500 (e.g. 5 for sanity check)
MAX_EPOCHS="${MAX_EPOCHS:-}"

# 2. Export strict nnUNet Environment Variables
# ------------------------------------------------------------------------------
export nnUNet_raw_data_base="$WORKSPACE_DIR/nnUNet_raw_data_base"
export nnUNet_preprocessed="$WORKSPACE_DIR/nnUNet_preprocessed"
# Results folder should point to Google Drive to persist checkpoints safely
export RESULTS_FOLDER="${RESULTS_FOLDER:-/content/drive/MyDrive/PI-CAI_Results}"

# Memory optimizations for Colab (prevents OOM Killer)
export nnUNet_n_proc_DA=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Ensure directories exist
mkdir -p "$nnUNet_raw_data_base"
mkdir -p "$nnUNet_preprocessed"
mkdir -p "$RESULTS_FOLDER"

# 2.5 Patch nnUNet with Z-SSMNet Custom Trainers
# ------------------------------------------------------------------------------
echo "=============================================================================="
echo "Patching native nnUNet installation with Z-SSMNet custom files..."
NNUNET_PKG=$(python -c 'import nnunet, os; print(os.path.dirname(nnunet.__file__))' 2>/dev/null)

# Colab Fallback for finding nnUNet package
if [ -z "$NNUNET_PKG" ] || [ ! -d "$NNUNET_PKG" ]; then
    NNUNET_PKG=$(find /usr/local/lib -type d -name "nnunet" 2>/dev/null | head -n 1)
fi

# Resolve Z-SSMNet path
if [ -d "/content/PI-CAI_Workspace/Z-SSMNet/src/z_ssmnet/z_nnmnet/training_docker" ]; then
    DOCKER_FILES="/content/PI-CAI_Workspace/Z-SSMNet/src/z_ssmnet/z_nnmnet/training_docker"
else
    DOCKER_FILES="$WORKSPACE_DIR/../Z-SSMNet/src/z_ssmnet/z_nnmnet/training_docker"
fi

if [ -d "$NNUNET_PKG" ] && [ -d "$DOCKER_FILES" ]; then
    cp "$DOCKER_FILES/nnUNetTrainerV2_focalLoss.py" "$NNUNET_PKG/training/network_training/nnUNet_variants/loss_function/nnUNetTrainerV2_focalLoss.py"
    cp "$DOCKER_FILES/MNet.py" "$NNUNET_PKG/network_architecture/MNet.py"
    cp "$DOCKER_FILES/MNet_basic_module.py" "$NNUNET_PKG/network_architecture/MNet_basic_module.py"
    cp "$DOCKER_FILES/MNet_myTrainer_zonal.py" "$NNUNET_PKG/training/network_training/MNet_myTrainer_zonal.py"
    cp "$DOCKER_FILES/dataset_loading.py" "$NNUNET_PKG/training/dataloading/dataset_loading.py"
    cp "$DOCKER_FILES/nnUNetTrainer.py" "$NNUNET_PKG/training/network_training/nnUNetTrainer.py"
    cp "$DOCKER_FILES/predict.py" "$NNUNET_PKG/inference/predict.py"
    cp "$DOCKER_FILES/run_training.py" "$NNUNET_PKG/run/run_training.py"
    echo "nnUNet successfully patched for Z-SSMNet."
else
    echo "ERROR: Could not find nnunet site-package or Z-SSMNet docker files to patch."
    echo "  NNUNET_PKG=$NNUNET_PKG"
    echo "  DOCKER_FILES=$DOCKER_FILES"
    exit 1
fi

echo "=============================================================================="
echo "Starting nnUNet Pipeline: $TASK_NAME"
echo "=============================================================================="
echo "nnUNet_raw_data_base: $nnUNet_raw_data_base"
echo "nnUNet_preprocessed : $nnUNet_preprocessed"
echo "RESULTS_FOLDER      : $RESULTS_FOLDER"
echo "MAX_CASES           : ${MAX_CASES:-all}"
echo "MAX_EPOCHS          : ${MAX_EPOCHS:-500 (default)}"
echo "=============================================================================="

# 3. Format Data for nnUNet
# ------------------------------------------------------------------------------
echo -e "\n>>> STEP 1: Formatting raw data to nnUNet structures..."
if [ -f "$WORKSPACE_DIR/.format_complete" ]; then
    echo "Data formatting already complete. Skipping..."
else
    MAX_CASES_ARG=""
    if [ -n "$MAX_CASES" ]; then
        MAX_CASES_ARG="--max_cases $MAX_CASES"
    fi
    python "$WORKSPACE_DIR/nnunet_prepare.py" \
        --source_dir "$SOURCE_DATA_DIR" \
        --nnunet_raw_dir "$nnUNet_raw_data_base/nnUNet_raw_data" \
        --splits_json "$CUSTOM_SPLITS_FILE" \
        $MAX_CASES_ARG
    touch "$WORKSPACE_DIR/.format_complete"
fi

# 4. Plan and Preprocess
# ------------------------------------------------------------------------------
echo -e "\n>>> STEP 2: Running nnUNet Plan & Preprocess..."
if [ -f "$WORKSPACE_DIR/.preprocess_complete" ]; then
    echo "Preprocessing already complete. Skipping..."
else
    # This command automatically figures out the dataset shape, cropping, and spacing.
    # NOTE: We intentionally omit --verify_dataset_integrity because the PI-CAI 
    # pre-processed dataset has harmless sub-voxel direction cosine differences 
    # between T2 and co-registered ADC/HBV images (floating-point rounding from 
    # the registration pipeline). nnUNet resamples everything anyway.
    nnUNet_plan_and_preprocess -t $TASK_ID
    touch "$WORKSPACE_DIR/.preprocess_complete"
fi

# 4.5 Convert splits.json → splits_final.pkl
# ------------------------------------------------------------------------------
echo -e "\n>>> STEP 2.5: Converting custom splits to nnUNet pickle format..."
PREPROCESSED_TASK_DIR="$nnUNet_preprocessed/$TASK_NAME"
SPLITS_PKL="$PREPROCESSED_TASK_DIR/splits_final.pkl"

if [ -f "$SPLITS_PKL" ]; then
    echo "splits_final.pkl already exists. Skipping..."
else
    if [ -f "$CUSTOM_SPLITS_FILE" ]; then
        python -c "
import json, pickle, numpy as np
from collections import OrderedDict
from pathlib import Path

splits_json = json.loads(Path('$CUSTOM_SPLITS_FILE').read_text())
preprocessed_dir = Path('$PREPROCESSED_TASK_DIR')

# Get actual case IDs that exist in the preprocessed cache
existing_cases = set()
for f in (preprocessed_dir / 'nnUNetData_plans_v2.1_stage0').glob('*.npz'):
    name = f.stem
    if '_seg' not in name:
        existing_cases.add(name)

print(f'Found {len(existing_cases)} preprocessed cases in cache.')

# Build the splits, filtering to only include cases that actually exist
splits = []
for fold in splits_json:
    train_keys = np.array([k for k in fold['train'] if k in existing_cases])
    val_keys = np.array([k for k in fold['val'] if k in existing_cases])
    splits.append(OrderedDict([('train', train_keys), ('val', val_keys)]))
    print(f'Fold: {len(train_keys)} train, {len(val_keys)} val')

with open('$SPLITS_PKL', 'wb') as f:
    pickle.dump(splits, f)

print(f'Successfully wrote splits_final.pkl with {len(splits)} folds.')
"
    else
        echo "WARNING: No custom splits.json found. nnUNet will generate its own random splits."
    fi
fi

# 5. Zonal Mask Injection
# ------------------------------------------------------------------------------
echo -e "\n>>> STEP 3: Injecting Zonal Masks into Preprocessed Cache..."
if [ -f "$WORKSPACE_DIR/.zonal_integration_complete" ]; then
    echo "Zonal integration already complete. Skipping..."
else
    python "$WORKSPACE_DIR/nnunet_zonal_integration.py" \
        --zonal_masks_dir "$SOURCE_DATA_DIR/zonal_masks" \
        --nnunet_preprocessed_dir "$nnUNet_preprocessed/$TASK_NAME/nnUNetData_plans_v2.1_stage0"
    touch "$WORKSPACE_DIR/.zonal_integration_complete"
fi

# 6. Train the Model
# ------------------------------------------------------------------------------
echo -e "\n>>> STEP 4: Training Fold 0..."
FOLD=0
TRAINER="myTrainer_zonal"
OUTPUT_DIR="$RESULTS_FOLDER/nnUNet/3d_fullres/$TASK_NAME/${TRAINER}__nnUNetPlansv2.1/fold_${FOLD}"

# Override max_num_epochs if MAX_EPOCHS is set (for sanity testing)
if [ -n "$MAX_EPOCHS" ]; then
    echo "Overriding max_num_epochs to $MAX_EPOCHS for sanity testing..."
    TRAINER_FILE="$NNUNET_PKG/training/network_training/MNet_myTrainer_zonal.py"
    if [ -f "$TRAINER_FILE" ]; then
        sed -i "s/self.max_num_epochs = 500/self.max_num_epochs = $MAX_EPOCHS/" "$TRAINER_FILE"
        echo "Patched max_num_epochs to $MAX_EPOCHS in $TRAINER_FILE"
    else
        echo "WARNING: Could not find trainer file to patch epochs: $TRAINER_FILE"
    fi
fi

if [ -f "$OUTPUT_DIR/model_latest.model" ]; then
    echo "Found existing checkpoint at $OUTPUT_DIR/model_latest.model"
    echo "Resuming training from checkpoint..."
    nnUNet_train 3d_fullres $TRAINER $TASK_ID $FOLD -c --npz
else
    echo "No checkpoint found. Starting training from scratch..."
    nnUNet_train 3d_fullres $TRAINER $TASK_ID $FOLD --npz
fi

echo "=============================================================================="
echo "Pipeline execution completed successfully."
echo "Model weights are saved in: $OUTPUT_DIR/"
echo "=============================================================================="
