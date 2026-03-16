"""Voxel-wise ensemble regression on saved NRF predictions.

Fits a per-voxel linear model: ground_truth = w1*pred1 + w2*pred2 + ... + bias
where each pred_i comes from a different fine-tuned NRF model.

Typical usage (ground truth from the saved H5 files):

    python voxel_regression.py \\
      --ensemble-subjects 1 5 7 \\
      --target-subject 2 \\
      --train-prediction-sources s1/ft_data.h5py s5/ft_data.h5py s7/ft_data.h5py \\
      --eval-prediction-sources  s1/best_epoch.h5py s5/best_epoch.h5py s7/best_epoch.h5py \\
      --train-ground-truth-h5 s1/ft_data.h5py \\
      --eval-ground-truth-h5  s1/best_epoch.h5py \\
      --roi nsdgeneral --fit-intercept --save-dir output/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LinearRegression

from utils.config import get_coordinate_path, get_image_order_path, get_avg_response_path


# ── CLI ──

def parse_args():
    p = argparse.ArgumentParser(description="Voxel-wise ensemble regression on saved NRF predictions.")
    p.add_argument("--ensemble-subjects", nargs="+", type=int, required=True,
                    help="Subject ids whose predictions are ensembled.")
    p.add_argument("--target-subject", type=int, required=True,
                    help="Subject to fit regression toward.")
    p.add_argument("--train-prediction-sources", nargs="+", required=True,
                    help="H5 files with train-split predictions, one per ensemble subject.")
    p.add_argument("--eval-prediction-sources", nargs="+", required=True,
                    help="H5 files with eval-split predictions, one per ensemble subject.")
    p.add_argument("--ground-truth-source", choices=["h5", "nsd"], default="h5",
                    help="'h5' (default): read ground truth from H5 files. 'nsd': read from NSD response files.")
    p.add_argument("--train-ground-truth-h5", type=str,
                    help="H5 file with train-split ground truth (required when --ground-truth-source h5).")
    p.add_argument("--eval-ground-truth-h5", type=str,
                    help="H5 file with eval-split ground truth (required when --ground-truth-source h5).")
    p.add_argument("--train-image-idx", type=str,
                    help=".npy file with training image ids. If omitted, all rows in the prediction files are used.")
    p.add_argument("--roi", nargs="+", type=str, required=True,
                    help="ROI name(s) to evaluate (e.g. nsdgeneral, FFA1 PPA V1v).")
    p.add_argument("--fit-intercept", action="store_true", default=False,
                    help="Fit a per-voxel bias term.")
    p.add_argument("--save-dir", type=str, required=True,
                    help="Directory to save outputs.")
    return p.parse_args()


# ── Data loading ──

def load_h5_dataset(path: str, name: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f[name])


def load_predictions(h5_paths, dataset_name, positions=None):
    """Load and stack predictions from multiple H5 files → (num_models, num_images, num_voxels)."""
    stack = []
    for path in h5_paths:
        data = load_h5_dataset(path, dataset_name)
        if positions is not None:
            data = data[positions]
        stack.append(data)
    return np.asarray(stack)


def load_ground_truth_nsd(subject, rois, image_ids):
    """Load ground truth responses from NSD averaged-response files."""
    response_order = np.load(get_image_order_path(subject))
    id_to_row = {int(img_id): idx for idx, img_id in enumerate(response_order)}
    rows = [id_to_row[int(i)] for i in image_ids]
    parts = [np.load(get_avg_response_path(subject, roi))[rows] for roi in rois]
    return np.hstack(parts)


def load_coordinates(subject, rois):
    """Load MNI coordinates for the target ROI(s)."""
    return np.vstack([np.load(get_coordinate_path(subject, r)) for r in rois])


def get_train_eval_image_ids(subject):
    """Split NSD image order into train (id >= 1000) and eval (id < 1000)."""
    order = np.load(get_image_order_path(subject))
    return order[order >= 1000], order[order < 1000]


# ── Regression ──

def correlation_by_voxel(target, prediction):
    n_voxels = target.shape[1]
    corr = np.zeros(n_voxels, dtype=np.float32)
    for v in range(n_voxels):
        x, y = target[:, v], prediction[:, v]
        if np.std(x) == 0 or np.std(y) == 0:
            corr[v] = 0.0
        else:
            corr[v] = np.corrcoef(x, y)[0, 1]
    return corr


def fit_voxel_regression(train_predictions, train_targets, fit_intercept):
    """Fit per-voxel linear regression from stacked model predictions to ground truth.

    Args:
        train_predictions: (num_models, num_images, num_voxels)
        train_targets:     (num_images, num_voxels)
    Returns:
        weights: (num_voxels, num_models)
        bias:    (num_voxels,)
        fitted:  (num_images, num_voxels) — training predictions from the fitted model
    """
    features = train_predictions.transpose(1, 0, 2)  # (num_images, num_models, num_voxels)
    n_voxels = features.shape[2]
    n_models = features.shape[1]

    weights = np.zeros((n_voxels, n_models), dtype=np.float32)
    bias = np.zeros(n_voxels, dtype=np.float32)
    fitted = np.zeros_like(train_targets, dtype=np.float32)

    for v in range(n_voxels):
        reg = LinearRegression(fit_intercept=fit_intercept)
        reg.fit(features[:, :, v], train_targets[:, v])
        fitted[:, v] = reg.predict(features[:, :, v])
        weights[v] = reg.coef_.astype(np.float32)
        bias[v] = float(reg.intercept_) if fit_intercept else 0.0

    return weights, bias, fitted


def predict_with_weights(prediction_stack, weights, bias):
    features = prediction_stack.transpose(1, 0, 2)
    return np.sum(features * weights.T[None, :, :], axis=1) + bias[None, :]


# ── Save ──

def save_results(save_dir, args, coordinates, train_image_ids, eval_image_ids,
                 weights, bias, train_corr, eval_corr):
    save_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(save_dir / "voxel_regression_results.h5py", "w") as f:
        f.create_dataset("weights", data=weights, dtype=np.float32)
        f.create_dataset("bias", data=bias, dtype=np.float32)
        f.create_dataset("train_corr", data=train_corr, dtype=np.float32)
        f.create_dataset("eval_corr", data=eval_corr, dtype=np.float32)
        f.create_dataset("coordinates", data=coordinates, dtype=np.float32)
        f.create_dataset("train_image_ids", data=np.asarray(train_image_ids, dtype=np.int32))
        f.create_dataset("eval_image_ids", data=np.asarray(eval_image_ids, dtype=np.int32))
        f.create_dataset("ensemble_subjects", data=np.asarray(args.ensemble_subjects, dtype=np.int32))
        f.attrs["metadata"] = json.dumps({
            "ensemble_subjects": args.ensemble_subjects,
            "target_subject": args.target_subject,
            "roi": args.roi,
            "fit_intercept": args.fit_intercept,
        }, sort_keys=True)

    np.save(save_dir / "weights_per_voxel.npy", weights)
    np.save(save_dir / "bias_per_voxel.npy", bias)
    np.save(save_dir / "eval_corr_per_voxel.npy", eval_corr)
    np.save(save_dir / "train_corr_per_voxel.npy", train_corr)


# ── Main ──

def main():
    args = parse_args()

    n = len(args.ensemble_subjects)
    if len(args.train_prediction_sources) != n:
        raise ValueError("--ensemble-subjects and --train-prediction-sources must have the same length.")
    if len(args.eval_prediction_sources) != n:
        raise ValueError("--ensemble-subjects and --eval-prediction-sources must have the same length.")

    target = args.target_subject
    pred_key = f"subj{target}_pred"
    gt_key = f"subj{target}_gt"

    train_image_ids, eval_image_ids = get_train_eval_image_ids(target)

    # Resolve which training rows to use
    if args.train_image_idx:
        selected_ids = np.load(args.train_image_idx).astype(np.int32)
        # If the prediction file already has the right number of rows, no subsetting needed
        with h5py.File(args.train_prediction_sources[0], "r") as f:
            n_rows = f[pred_key].shape[0]
        if n_rows == len(selected_ids):
            positions = None
        else:
            id_to_pos = {int(img_id): idx for idx, img_id in enumerate(train_image_ids)}
            positions = np.array([id_to_pos[int(i)] for i in selected_ids], dtype=np.int32)
        train_image_ids = selected_ids
    else:
        positions = None  # use all rows from prediction files

    # Load predictions: (num_models, num_images, num_voxels)
    train_preds = load_predictions(args.train_prediction_sources, pred_key, positions)
    eval_preds = load_predictions(args.eval_prediction_sources, pred_key)

    # Load ground truth: (num_images, num_voxels)
    if args.ground_truth_source == "h5":
        if not args.train_ground_truth_h5 or not args.eval_ground_truth_h5:
            raise ValueError("--train-ground-truth-h5 and --eval-ground-truth-h5 required with --ground-truth-source h5.")
        train_gt = load_h5_dataset(args.train_ground_truth_h5, gt_key)
        eval_gt = load_h5_dataset(args.eval_ground_truth_h5, gt_key)
        if positions is not None:
            train_gt = train_gt[positions]
    else:
        train_gt = load_ground_truth_nsd(target, args.roi, train_image_ids)
        eval_gt = load_ground_truth_nsd(target, args.roi, eval_image_ids)

    # Validate coordinate / voxel count match
    coordinates = load_coordinates(target, args.roi)
    if coordinates.shape[0] != train_gt.shape[1]:
        raise ValueError(
            f"Coordinate count {coordinates.shape[0]} != voxel count {train_gt.shape[1]} for ROI '{args.roi}'."
        )

    # Fit per-voxel regression and evaluate
    weights, bias, train_fitted = fit_voxel_regression(train_preds, train_gt, args.fit_intercept)
    eval_fitted = predict_with_weights(eval_preds, weights, bias)

    train_corr = correlation_by_voxel(train_gt, train_fitted)
    eval_corr = correlation_by_voxel(eval_gt, eval_fitted)

    save_results(Path(args.save_dir), args, coordinates, train_image_ids, eval_image_ids,
                 weights, bias, train_corr, eval_corr)

    print(f"Saved to {args.save_dir}")
    print(f"Median train corr: {float(np.median(train_corr)):.6f}")
    print(f"Median eval corr:  {float(np.median(eval_corr)):.6f}")


if __name__ == "__main__":
    main()
