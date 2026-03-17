## NSD Preprocessing

These scripts convert raw NSD beta sessions into per-ROI response files that the training pipeline reads.

### Downloading stimulus images

The training dataloader also requires pre-resized 227x227 NSD stimulus images
(`S{subject}_stimuli_227.h5py`). These are not produced by the preprocessing
scripts but are needed at training time. Download them from Hugging Face:

```bash
pip install huggingface_hub
huggingface-cli download haomiao8/NRF-stimuli --repo-type dataset --local-dir /path/to/stimuli
```

Then set `stimuli_dir` in `config.yaml` to the download directory.

### Path configuration

NSD input paths and the output directory are configured in `config.yaml`:

```yaml
paths:
  # NSD raw input — each *_dir contains subjXX/ subdirectories
  brain_mask_dir: "/path/to/nsddata/ppdata"         # subjXX/func1pt8mm/brainmask.nii.gz
  roi_label_dir:  "/path/to/nsddata/ppdata"         # subjXX/func1pt8mm/roi/*.nii.gz
  transforms_dir: "/path/to/nsddata/ppdata"         # subjXX/transforms/MNI-to-func1pt8.nii.gz
  betas_dir:      "/path/to/nsddata_betas/ppdata"   # subjXX/func1pt8mm/betas_fithrf_GLMdenoise_RR/
  expdesign_mat:  "/path/to/nsd_expdesign.mat"      # single file

  data_root: "/path/to/processed_nsd/"              # output directory
```

For a standard NSD download, `brain_mask_dir`, `roi_label_dir`, and `transforms_dir` all point
to the same `ppdata/` directory. If you have custom preprocessing, point each to wherever those
files live.

All outputs are written under:
`data_root/roi_masks/subjXX/`, `data_root/roi_response/subjXX/`,
`data_root/roi_response_average/subjXX/`, and `data_root/MNI_coordinate/subjXX/`.

### Quick start

**Before running**, make sure the paths in `config.yaml` point to where your NSD data is stored.
Then preprocessing is just:

```bash
python -m nsd_data_processing.getROImask --subject 1
python -m nsd_data_processing.getmaskedROI --subject 1
python -m nsd_data_processing.getmaskedROIaverage --subject 1
```

Step 1 automatically uses the subject's ROI list from `config.yaml` (under `subject_rois`).

### Output layout

Everything is written under `data_root/<dirname>/subjXX/`.

Step 1 outputs:

- `roi_masks/subjXX/roi_names.json`
- `roi_masks/subjXX/<roi>_mask.npy`
- `MNI_coordinate/subjXX/<roi>_MNI_coordinate.npy`

Step 2 outputs:

- `roi_response/subjXX/roi_names.json`
- `roi_response/subjXX/trial_image_ids.npy`
- `roi_response/subjXX/<roi>.npy`

Step 3 outputs:

- `roi_response_average/subjXX/roi_names.json`
- `roi_response_average/subjXX/image_order.npy`
- `roi_response_average/subjXX/<roi>.npy`

### NSD files read (by config key)

**Step 1** reads from:

- `brain_mask_dir` — `subjXX/func1pt8mm/brainmask.nii.gz`
- `roi_label_dir` — `subjXX/func1pt8mm/roi/nsdgeneral.nii.gz`, `floc-faces.nii.gz`, `floc-bodies.nii.gz`, `floc-places.nii.gz`, `floc-words.nii.gz`, `prf-visualrois.nii.gz`
- `transforms_dir` — `subjXX/transforms/MNI-to-func1pt8.nii.gz`

**Step 2** additionally reads from:

- `betas_dir` — `subjXX/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session*.nii.gz`
- `expdesign_mat` — direct path to `nsd_expdesign.mat`

### Optional arguments

```bash
# Step 1: override which ROIs to extract
python -m nsd_data_processing.getROImask --subject 1 --roi FFA1 PPA V1v

# Step 2: limit sessions, skip z-scoring, or process a subset of ROIs
python -m nsd_data_processing.getmaskedROI --subject 1 --session-limit 10 --no-zscore --roi FFA1 PPA

# Step 3: average only specific ROIs
python -m nsd_data_processing.getmaskedROIaverage --subject 1 --roi FFA1 PPA
```
