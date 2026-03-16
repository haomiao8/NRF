#!/bin/bash
#SBATCH -J nrf_test
#SBATCH -t 0-06:00:00
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH -o test_pipeline_%j.out

# End-to-end pipeline test: preprocessing → training → fine-tuning → voxel regression
# Usage: sbatch test_job.sh
source activate neurogen
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
else
  echo "Unable to locate conda activation script. Set CONDA_BASE or ensure 'conda' is on PATH."
  exit 1
fi
conda activate "${CONDA_ENV_NAME:-neurogen}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR"
# Ensure local modules (e.g., utils.config) are importable in batch environments.
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

SUBJECT=1
NSD_ROOT="$(python -c 'from utils.config import NSD_ROOT; print(NSD_ROOT)')"
SAVE_ROOT="$(python -c 'from utils.config import SAVE_ROOT; print(SAVE_ROOT)')"
RUN_NAME="test_pipeline"

# echo "============================================="
# echo "NRF End-to-End Pipeline Test"
# echo "  Subject:   $SUBJECT"
# echo "  NSD root:  $NSD_ROOT"
# echo "  Save root: $SAVE_ROOT"
# echo "  Run name:  $RUN_NAME"
# echo "============================================="

# # ── Step 1/6: Extract ROI masks and MNI coordinates ──
# echo ""
# echo "[Step 1/6] Extracting ROI masks..."
# python -B -m nsd_data_processing.getROImask --subject "$SUBJECT"
# echo "[Step 1/6] Done."

# # ── Step 2/6: Apply masks to beta sessions, save trial-level responses ──
# echo ""
# echo "[Step 2/6] Saving trial-level masked ROI responses..."
# python -B -m nsd_data_processing.getmaskedROI --subject "$SUBJECT"
# echo "[Step 2/6] Done."

# # ── Step 3/6: Average repeated trials by image id ──
# echo ""
# echo "[Step 3/6] Averaging repeated trials..."
# python -B -m nsd_data_processing.getmaskedROIaverage --subject "$SUBJECT"
# echo "[Step 3/6] Done."

# # ── Step 4/6: Train base model (1 epoch) ──
# echo ""
# echo "[Step 4/6] Training base model (1 epoch)..."
# python -B main_train.py \
#   --exp_name "${RUN_NAME}/base_s${SUBJECT}" \
#   --exp_config_dir training_config.yaml \
#   --data_subject_list "$SUBJECT" \
#   --model_subject_list "$SUBJECT" \
#   --roi_list nsdgeneral \
#   --epochs 1 \
#   --batch_size 32 \
#   --mode train
# echo "[Step 4/6] Done."

# ── Step 5/6: Fine-tune from base model (1 epoch) ──
# echo ""
# echo "[Step 5/6] Fine-tuning from base model (1 epoch)..."
# python -B main_train.py \
#   --exp_name "${RUN_NAME}/ft_s${SUBJECT}_to_s${SUBJECT}" \
#   --pretrained_config_dir "${RUN_NAME}/base_s${SUBJECT}" \
#   --pretrained_ckpt_name best_model \
#   --data_subject_list "$SUBJECT" \
#   --roi_list nsdgeneral \
#   --epochs 1 \
#   --batch_size 32 \
#   --mode finetune \
#   --train_encoder \
#   --train_feature_merger
#   --
# echo "[Step 5/6] Done."

# ── Step 6/7: Voxel regression ──
# echo ""
# echo "[Step 6/7] Running voxel regression..."
# FT_DIR="${SAVE_ROOT}/${RUN_NAME}/ft_s${SUBJECT}_to_s${SUBJECT}"
# python -B voxel_regression.py \
#   --ensemble-subjects "$SUBJECT" \
#   --target-subject "$SUBJECT" \
#   --train-prediction-sources "${FT_DIR}/ft_data.h5py" \
#   --eval-prediction-sources "${FT_DIR}/best_epoch.h5py" \
#   --ground-truth-source h5 \
#   --train-ground-truth-h5 "${FT_DIR}/ft_data.h5py" \
#   --eval-ground-truth-h5 "${FT_DIR}/best_epoch.h5py" \
#   --roi nsdgeneral \
#   --fit-intercept \
#   --save-dir "${SAVE_ROOT}/${RUN_NAME}/voxel_regression"
# echo "[Step 6/7] Done."

# ── Step 7/7: Evaluate the fine-tuned model ──
echo ""
echo "[Step 7/7] Evaluating fine-tuned model on validation set..."
python -B main_train.py \
  --exp_name "${RUN_NAME}/eval_s${SUBJECT}" \
  --exp_config_dir "${FT_DIR}/experiment_config.yaml" \
  --data_subject_list "$SUBJECT" \
  --eval_subject_list "$SUBJECT" \
  --roi_list nsdgeneral \
  --mode evaluate \
  --pretrained_config_dir "${RUN_NAME}/ft_s${SUBJECT}_to_s${SUBJECT}" \
  --pretrained_ckpt_name best_model \
  --eval_output_name eval_results
echo "[Step 7/7] Done."

echo ""
echo "============================================="
echo "All steps completed successfully!"
echo "Results saved under: ${SAVE_ROOT}/${RUN_NAME}/"
echo "  Evaluation output: ${SAVE_ROOT}/${RUN_NAME}/eval_s${SUBJECT}/eval_results.h5py"
echo "============================================="
