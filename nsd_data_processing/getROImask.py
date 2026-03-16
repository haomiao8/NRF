from __future__ import annotations

import argparse

from nsd_data_processing.common import (
    build_roi_masks,
    get_subject_rois,
    load_brain_mask,
    mask_dir,
    save_roi_masks,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: save brain-space ROI masks and MNI coordinates."
    )
    parser.add_argument("--subject", type=int, required=True, help="NSD subject id, e.g. 1.")
    parser.add_argument(
        "--roi",
        nargs="*",
        default=None,
        help="ROI names (e.g. FFA1 PPA V1v). Defaults to subject's ROIs from config.yaml.",
    )
    args = parser.parse_args()

    roi_names = args.roi if args.roi is not None else get_subject_rois(args.subject)

    brain_mask_full = load_brain_mask(args.subject)
    roi_masks_full = build_roi_masks(args.subject, brain_mask_full, roi_names)
    save_roi_masks(args.subject, brain_mask_full, roi_masks_full)

    print(f"Saved {len(roi_masks_full)} ROI masks to {mask_dir(args.subject)}")
    for name, roi_mask in roi_masks_full.items():
        print(f"  {name}: {int(roi_mask.sum())} voxels")


if __name__ == "__main__":
    main()
