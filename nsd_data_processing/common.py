from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import nibabel as nib
import numpy as np
from scipy.io import loadmat

from utils.config import PATHS, SUBJECT_ROIS

# ── Path helpers ──
# All raw NSD data is read from nsd_root, all processed output goes to data_root.

NSD_ROOT = Path(PATHS["nsd_root"])
DATA_ROOT = Path(PATHS["data_root"])

ROI_MANIFEST = "roi_names.json"
TRIAL_IMAGE_IDS = "trial_image_ids.npy"
IMAGE_ORDER = PATHS.get("image_order_filename", "image_order.npy")


def _subj(subject: int) -> str:
    return f"subj{subject:02d}"


# -- Input paths (raw NSD data under nsd_root) --

def betas_dir(subject: int) -> Path:
    return NSD_ROOT / "responses" / _subj(subject) / "func1pt8mm" / "betas_fithrf_GLMdenoise_RR"


def brain_mask_path(subject: int) -> Path:
    return NSD_ROOT / "mask" / "ppdata" / _subj(subject) / "func1pt8mm" / "brainmask_inflated_1.0.nii"


def nsdgeneral_mask_path(subject: int) -> Path:
    return NSD_ROOT / "roimask" / _subj(subject) / "nsdgeneral.nii.gz"


def roi_label_dir(subject: int) -> Path:
    return NSD_ROOT / "mask" / "ppdata" / _subj(subject) / "func1pt8mm" / "roi"


def transform_path(subject: int) -> Path:
    return NSD_ROOT / "transforms" / "ppdata" / _subj(subject) / "MNI-to-func1pt8.nii.gz"


def ordering_path() -> Path:
    return NSD_ROOT / "experiments" / "nsd_expdesign.mat"


# -- Output paths (processed data under data_root) --




def mask_dir(subject: int) -> Path:
    return DATA_ROOT / Path(PATHS.get("roi_masks_dirname", "roi_masks")) / _subj(subject)


def coordinate_dir(subject: int) -> Path:
    return DATA_ROOT / Path(PATHS.get("coordinate_dirname", "MNI_coordinate")) / _subj(subject)


def roi_response_dir(subject: int) -> Path:
    return DATA_ROOT / Path(PATHS.get("roi_response_dirname", "roi_response")) / _subj(subject)


def roi_response_average_dir(subject: int) -> Path:
    return DATA_ROOT / Path(PATHS.get("avg_response_dirname", "roi_response_average")) / _subj(subject)


def get_subject_rois(subject: int) -> list[str]:
    rois = SUBJECT_ROIS.get(subject)
    if not rois:
        raise ValueError(f"No ROIs configured for subject {subject} in config.yaml")
    return list(rois)


# ── NSD ROI label definitions ──

MNI_INDEX_TO_MM = np.array(
    [
        [-1.0, 0.0, 0.0, 90.0],
        [0.0, 1.0, 0.0, -126.0],
        [0.0, 0.0, 1.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

NSD_ROI_LABELS: Dict[str, tuple[str, int]] = {
    "OFA": ("floc-faces.nii.gz", 1),
    "FFA1": ("floc-faces.nii.gz", 2),
    "FFA2": ("floc-faces.nii.gz", 3),
    "mTLfaces": ("floc-faces.nii.gz", 4),
    "aTLfaces": ("floc-faces.nii.gz", 5),
    "EBA": ("floc-bodies.nii.gz", 1),
    "FBA1": ("floc-bodies.nii.gz", 2),
    "FBA2": ("floc-bodies.nii.gz", 3),
    "mTLbodies": ("floc-bodies.nii.gz", 4),
    "OPA": ("floc-places.nii.gz", 1),
    "PPA": ("floc-places.nii.gz", 2),
    "RSC": ("floc-places.nii.gz", 3),
    "OWFA": ("floc-words.nii.gz", 1),
    "VWFA1": ("floc-words.nii.gz", 2),
    "VWFA2": ("floc-words.nii.gz", 3),
    "mfswords": ("floc-words.nii.gz", 4),
    "mTLwords": ("floc-words.nii.gz", 5),
    "V1v": ("prf-visualrois.nii.gz", 1),
    "V1d": ("prf-visualrois.nii.gz", 2),
    "V2v": ("prf-visualrois.nii.gz", 3),
    "V2d": ("prf-visualrois.nii.gz", 4),
    "V3v": ("prf-visualrois.nii.gz", 5),
    "V3d": ("prf-visualrois.nii.gz", 6),
    "hV4": ("prf-visualrois.nii.gz", 7),
}


# ── Volume / mask I/O ──

def load_volume(path: Path) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata())


def load_brain_mask(subject: int) -> np.ndarray:
    return load_volume(brain_mask_path(subject)) > 0


def normalize_nsd_roi_names(roi_names: Iterable[str]) -> list[str]:
    normalized = [r.strip() for r in roi_names if r.strip()]
    if not normalized:
        return []
    if any(r.lower() == "all" for r in normalized):
        return list(NSD_ROI_LABELS.keys())
    # nsdgeneral is always included separately, so skip it here
    normalized = [r for r in normalized if r != "nsdgeneral"]
    unknown = [r for r in normalized if r not in NSD_ROI_LABELS]
    if unknown:
        raise ValueError(f"Unknown NSD ROI names: {unknown}. Available: {sorted(NSD_ROI_LABELS)}")
    return normalized


# ── Beta / trial loading ──

def iter_beta_files(subject: int, session_limit: int | None = None) -> list[Path]:
    beta_files = sorted(betas_dir(subject).glob("betas_session*.nii.gz"))
    if session_limit is not None:
        beta_files = beta_files[:session_limit]
    if not beta_files:
        raise FileNotFoundError(f"No beta session files found under {betas_dir(subject)}.")
    return beta_files


def load_beta_file(beta_path: Path, brain_mask_flat: np.ndarray) -> np.ndarray:
    session = np.asarray(nib.load(beta_path).dataobj, dtype=np.float32)
    session = session.transpose((3, 0, 1, 2)).reshape(session.shape[-1], -1)
    return session[:, brain_mask_flat] / 300.0


def load_trial_image_ids(num_trials: int) -> np.ndarray:
    path = ordering_path()
    if path.suffix == ".mat":
        ordering = loadmat(str(path))["masterordering"].reshape(-1) - 1
        if num_trials > len(ordering):
            raise ValueError(f"Requested {num_trials} trials, but ordering file only has {len(ordering)} entries.")
        return ordering[:num_trials].astype(np.int32)
    ordering = np.load(path).astype(np.int32)
    if len(ordering) != num_trials:
        raise ValueError(f"Ordering file {path} has length {len(ordering)}, expected {num_trials} trial entries.")
    return ordering


def zscore_trials(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=0, keepdims=True)
    std = np.std(values, axis=0, keepdims=True)
    return np.nan_to_num((values - mean) / (std + 1e-6)).astype(np.float32)


def average_responses_by_image(trial_responses: np.ndarray, trial_image_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_image_ids = np.unique(trial_image_ids)
    averaged = np.vstack([trial_responses[trial_image_ids == img_id].mean(axis=0) for img_id in unique_image_ids])
    return averaged.astype(np.float32), unique_image_ids.astype(np.int32)


# ── ROI mask building & saving ──

def extract_mask_coordinates(subject: int, mask_full: np.ndarray) -> np.ndarray:
    transform = load_volume(transform_path(subject)).astype(np.float32)
    coords = transform[mask_full]
    coords[coords == 9999] = np.nan
    coords = coords - 1.0
    coords_h = np.concatenate([coords, np.ones((coords.shape[0], 1), dtype=np.float32)], axis=1)
    return (MNI_INDEX_TO_MM @ coords_h.T)[:3].T.astype(np.float32)


def build_roi_masks(
    subject: int,
    brain_mask_full: np.ndarray,
    roi_names: Iterable[str],
) -> dict[str, np.ndarray]:
    nsdgen_mask = load_volume(nsdgeneral_mask_path(subject)) > 0
    if nsdgen_mask.shape != brain_mask_full.shape:
        raise ValueError(
            f"nsdgeneral mask shape {nsdgen_mask.shape} != brain mask shape {brain_mask_full.shape}."
        )

    roi_masks: dict[str, np.ndarray] = {"nsdgeneral": nsdgen_mask & brain_mask_full}
    label_volumes: dict[str, np.ndarray] = {}
    roi_dir = roi_label_dir(subject)

    for name in normalize_nsd_roi_names(roi_names):
        volume_name, label_value = NSD_ROI_LABELS[name]
        if volume_name not in label_volumes:
            label_volumes[volume_name] = load_volume(roi_dir / volume_name)
        roi_masks[name] = (label_volumes[volume_name] == label_value) & brain_mask_full

    return roi_masks


def save_roi_names(output_dir: Path, roi_names: Iterable[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / ROI_MANIFEST, "w", encoding="ascii") as f:
        json.dump(list(roi_names), f, indent=2)


def load_roi_names(output_dir: Path, suffix: str) -> list[str]:
    manifest = output_dir / ROI_MANIFEST
    if manifest.exists():
        with open(manifest, "r", encoding="ascii") as f:
            return list(json.load(f))
    return sorted(p.name[: -len(suffix)] for p in output_dir.glob(f"*{suffix}"))


def save_roi_masks(
    subject: int,
    brain_mask_full: np.ndarray,
    roi_masks_full: dict[str, np.ndarray],
) -> None:
    out_mask_dir = mask_dir(subject)
    out_coord_dir = coordinate_dir(subject)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_coord_dir.mkdir(parents=True, exist_ok=True)
    brain_mask_flat = brain_mask_full.reshape(-1)

    for name, roi_mask_full in roi_masks_full.items():
        roi_mask = roi_mask_full.reshape(-1)[brain_mask_flat].astype(bool)
        coords = extract_mask_coordinates(subject, roi_mask_full)
        if coords.shape[0] != int(roi_mask.sum()):
            raise ValueError(f"Coordinate count mismatch for ROI '{name}'.")
        np.save(out_mask_dir / f"{name}_mask.npy", roi_mask)
        np.save(out_coord_dir / f"{name}_MNI_coordinate.npy", coords.astype(np.float32))

    save_roi_names(out_mask_dir, roi_masks_full.keys())


def load_saved_roi_mask(subject: int, roi_name: str) -> np.ndarray:
    return np.load(mask_dir(subject) / f"{roi_name}_mask.npy").astype(bool)
