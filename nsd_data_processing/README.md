## NSD Preprocessing

These scripts convert raw NSD beta sessions into per-ROI response files that the training pipeline reads.

### Path configuration

Two directories are involved:

- **`nsd_root`** — where the raw NSD dataset lives (read-only input, including betas).
- **`data_root`** — where processed outputs are saved. The training dataloader reads from this same directory.

Set both in `config.yaml`:

```yaml
paths:
  nsd_root: "/path/to/nsd_data/"
  data_root: "/path/to/processed_nsd/"
```

All outputs are written under `data_root/subjXX/`.

### Quick start

Once `config.yaml` is set, preprocessing is just:

```bash
python -m nsd_data_processing.getROImask --subject 1 --roi 
python -m nsd_data_processing.getmaskedROI --subject 1 --roi 
python -m nsd_data_processing.getmaskedROIaverage --subject 1 --roi 
```

Step 1 automatically uses the subject's ROI list from `config.yaml` (under `subject_rois`).

### Output layout

Everything is written under `data_root/subjXX/`.

Step 1 outputs:

- `roi_masks/roi_names.json`
- `roi_masks/<roi>_mask.npy`
- `MNI_coordinate/<roi>_MNI_coordinate.npy`

Step 2 outputs:

- `roi_response/roi_names.json`
- `roi_response/trial_image_ids.npy`
- `roi_response/<roi>.npy`

Step 3 outputs:

- `roi_response_average/roi_names.json`
- `roi_response_average/image_order.npy`
- `roi_response_average/<roi>.npy`

### NSD files read from `nsd_root`

Step 1 reads:

- `mask/ppdata/subjXX/func1pt8mm/brainmask_inflated_1.0.nii`
- `roimask/subjXX/nsdgeneral.nii.gz`
- `transforms/ppdata/subjXX/MNI-to-func1pt8.nii.gz`
- `mask/ppdata/subjXX/func1pt8mm/roi/` (ROI label volumes)

Step 2 additionally reads:

- `responses/subjXX/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session*.nii.gz`
- `experiments/nsd_expdesign.mat`

### Optional arguments

```bash
# Step 1: override which ROIs to extract
python -m nsd_data_processing.getROImask --subject 1 --nsd-roi FFA1 PPA V1v

# Step 2: limit sessions, skip z-scoring, or process a subset of ROIs
python -m nsd_data_processing.getmaskedROI --subject 1 --session-limit 10 --no-zscore --roi FFA1 PPA

# Step 3: average only specific ROIs
python -m nsd_data_processing.getmaskedROIaverage --subject 1 --roi FFA1 PPA
```
