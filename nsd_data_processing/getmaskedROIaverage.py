from __future__ import annotations

import argparse

import numpy as np

from nsd_data_processing.common import (
    IMAGE_ORDER,
    TRIAL_IMAGE_IDS,
    average_responses_by_image,
    load_roi_names,
    roi_response_average_dir,
    roi_response_dir,
    save_roi_names,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: average repeated trial-level ROI responses by image id."
    )
    parser.add_argument("--subject", type=int, required=True, help="NSD subject id, e.g. 1.")
    parser.add_argument("--roi", nargs="*", default=[], help="Optional ROI subset. Defaults to all ROIs from step 2.")
    args = parser.parse_args()

    subject = args.subject
    in_dir = roi_response_dir(subject)
    out_dir = roi_response_average_dir(subject)

    trial_image_ids = np.load(in_dir / TRIAL_IMAGE_IDS).astype(np.int32)
    roi_names = args.roi if args.roi else load_roi_names(in_dir, suffix=".npy")
    roi_names = [name for name in roi_names if name != TRIAL_IMAGE_IDS[: -len(".npy")]]

    out_dir.mkdir(parents=True, exist_ok=True)
    save_roi_names(out_dir, roi_names)

    image_order = None
    for name in roi_names:
        roi_trials = np.load(in_dir / f"{name}.npy").astype(np.float32)
        if roi_trials.shape[0] != len(trial_image_ids):
            raise ValueError(
                f"ROI '{name}' has {roi_trials.shape[0]} trials, but {len(trial_image_ids)} image ids were saved."
            )
        roi_mean, current_order = average_responses_by_image(roi_trials, trial_image_ids)
        np.save(out_dir / f"{name}.npy", roi_mean.astype(np.float32))
        if image_order is None:
            image_order = current_order
        elif not np.array_equal(image_order, current_order):
            raise ValueError(f"Image order mismatch while averaging ROI '{name}'.")
        print(f"{name}: images={roi_mean.shape[0]}, voxels={roi_mean.shape[1]}")

    if image_order is None:
        raise ValueError("No ROI responses were found to average.")
    np.save(out_dir / IMAGE_ORDER, image_order.astype(np.int32))
    print(f"Saved averaged ROI responses to {out_dir}")


if __name__ == "__main__":
    main()
