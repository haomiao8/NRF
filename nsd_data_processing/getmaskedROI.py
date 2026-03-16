from __future__ import annotations

import argparse

import numpy as np

from nsd_data_processing.common import (
    TRIAL_IMAGE_IDS,
    iter_beta_files,
    load_beta_file,
    load_brain_mask,
    load_roi_names,
    load_saved_roi_mask,
    load_trial_image_ids,
    mask_dir,
    roi_response_dir,
    save_roi_names,
    zscore_trials,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: apply saved ROI masks to beta sessions and save trial-level ROI responses."
    )
    parser.add_argument("--subject", type=int, required=True, help="NSD subject id, e.g. 1.")
    parser.add_argument("--roi", nargs="*", default=[], help="Optional ROI subset. Defaults to all masks from step 1.")
    parser.add_argument("--session-limit", type=int, help="Only process the first N beta sessions.")
    parser.add_argument("--no-zscore", action="store_true", help="Keep responses in raw scaled beta units.")
    args = parser.parse_args()

    subject = args.subject
    brain_mask_full = load_brain_mask(subject)
    brain_mask_flat = brain_mask_full.reshape(-1)

    roi_names = args.roi if args.roi else load_roi_names(mask_dir(subject), suffix="_mask.npy")
    roi_masks = {name: load_saved_roi_mask(subject, name) for name in roi_names}

    expected_voxels = int(brain_mask_full.sum())
    for name, mask in roi_masks.items():
        if mask.shape[0] != expected_voxels:
            raise ValueError(f"Mask '{name}' has {mask.shape[0]} voxels, expected {expected_voxels}.")

    roi_chunks = {name: [] for name in roi_names}
    total_trials = 0
    for beta_file in iter_beta_files(subject, args.session_limit):
        session = load_beta_file(beta_file, brain_mask_flat)
        total_trials += session.shape[0]
        for name, mask in roi_masks.items():
            roi_session = session[:, mask].astype(np.float32)
            if not args.no_zscore:
                roi_session = zscore_trials(roi_session)
            roi_chunks[name].append(roi_session)

    trial_image_ids = load_trial_image_ids(total_trials)

    out_dir = roi_response_dir(subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_roi_names(out_dir, roi_names)
    np.save(out_dir / TRIAL_IMAGE_IDS, trial_image_ids.astype(np.int32))

    for name in roi_names:
        roi_trials = np.concatenate(roi_chunks[name], axis=0).astype(np.float32)
        np.save(out_dir / f"{name}.npy", roi_trials)
        print(f"{name}: trials={roi_trials.shape[0]}, voxels={roi_trials.shape[1]}")

    print(f"Saved trial-level ROI responses to {out_dir}")


if __name__ == "__main__":
    main()
