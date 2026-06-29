#!/bin/bash
set -e

# ==============================================================================
# PI-CAI Z-SSMNet nnUNet Pipeline Orchestrator for DGX Servers
# ==============================================================================

# 1. Define Paths 
# ------------------------------------------------------------------------------
# These are expected to be set by the Colab Notebook environment
WORKSPACE_DIR="${WORKSPACE_DIR:-/content/PI-CAI_Workspace}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-/content/drive/MyDrive/PI-CAI_pre-processed}"
CUSTOM_SPLITS_FILE="$WORKSPACE_DIR/splits.json"
TASK_NAME="Task2302_z-nnmnet"
TASK_ID="2302"

# 2. Export strict nnUNet Environment Variables
# ------------------------------------------------------------------------------
export nnUNet_raw_data_base="$WORKSPACE_DIR/nnUNet_raw_data_base"
export nnUNet_preprocessed="$WORKSPACE_DIR/nnUNet_preprocessed"
# Results folder should point to Google Drive to save checkpoints safely
export RESULTS_FOLDER="${RESULTS_FOLDER:-/content/drive/MyDrive/PI-CAI_Results}"

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
    echo "WARNING: Could not find nnunet site-package or Z-SSMNet docker files to patch."
fi

echo "=============================================================================="
echo "Starting nnUNet Pipeline: $TASK_NAME"
echo "=============================================================================="
echo "nnUNet_raw_data_base: $nnUNet_raw_data_base"
echo "nnUNet_preprocessed : $nnUNet_preprocessed"
echo "RESULTS_FOLDER      : $RESULTS_FOLDER"
echo "=============================================================================="

# 3. Format Data for nnUNet
# ------------------------------------------------------------------------------
echo -e "\n>>> STEP 1: Formatting raw data to nnUNet structures..."
# Run the python script to explicitly copy images and generate dataset.json
python "$WORKSPACE_DIR/nnunet_prepare.py" \
    --source_dir "$SOURCE_DATA_DIR" \
    --nnunet_raw_dir "$nnUNet_raw_data_base/nnUNet_raw_data" \
    --splits_json "$CUSTOM_SPLITS_FILE"

# 4. Plan and Preprocess
# ------------------------------------------------------------------------------
echo -e "\n>>> STEP 2: Running nnUNet Plan & Preprocess..."
# This command automatically figures out the dataset shape, cropping, and spacing.
nnUNet_plan_and_preprocess -t $TASK_ID --verify_dataset_integrity

# 5. Zonal Mask Injection
# ------------------------------------------------------------------------------
echo -e "\n>>> STEP 3: Injecting Zonal Masks into Preprocessed Cache..."
# The custom trainer (myTrainer_zonal) expects the zonal masks to be dynamically
# cropped and cached as .npz files just like the MRI images.
python "$WORKSPACE_DIR/nnunet_zonal_integration.py" \
    --zonal_masks_dir "$SOURCE_DATA_DIR/zonal_masks" \
    --nnunet_preprocessed_dir "$nnUNet_preprocessed/$TASK_NAME/nnUNetData_plans_v2.1_stage0"

# 6. Train the Model
# ------------------------------------------------------------------------------
echo -e "\n>>> STEP 4: Training Fold 0..."
FOLD=0
TRAINER="myTrainer_zonal"
OUTPUT_DIR="$RESULTS_FOLDER/nnUNet/3d_fullres/$TASK_NAME/${TRAINER}__nnUNetPlansv2.1/fold_${FOLD}"

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
