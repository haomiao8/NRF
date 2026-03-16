from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional, Sequence

import h5py
import numpy as np
import torch

from utils.config import PATHS, SUBJECT_ROIS
from utils.config import get_coordinate_path, get_image_order_path, get_avg_response_path


nsd_root = Path(PATHS["nsd_root"])
stim_root = nsd_root / "stimuli"
responses_root = Path(PATHS["data_root"])



def get_subject_rois(subject_id: int):
    return SUBJECT_ROIS[int(subject_id)]


class neural_loader(torch.utils.data.Dataset):
    def __init__(
        self,
        subject_id,
        selected_rois: Optional[Sequence[str]] = None,
        trn_image_idx=None,
        shuffle_coord: bool = False,
        subsample_voxels: bool = False,
        selected_layers: Sequence[int] = (3, 6),
        num_voxels: Optional[int] = None,
    ):
        self.subject_id = subject_id if isinstance(subject_id, list) else [subject_id]
        self.selected_layers = list(selected_layers)
        self.shuffle_coord = shuffle_coord
        self.subsample_voxels = subsample_voxels
        self.num_voxels = num_voxels
        self.num_stimulus: Dict[str, int] = {}
        self.neural_sizes: Dict[str, int] = {}
        self.trn_neural: Dict[str, np.ndarray] = {}
        self.val_neural: Dict[str, np.ndarray] = {}
        self.trn_num_images: Dict[str, int] = {}
        self.val_num_images: Dict[str, int] = {}
        self.mni_coordinates: Dict[str, np.ndarray] = {}
        self.trn_image_ids: Dict[str, np.ndarray] = {}
        self.val_image_ids: Dict[str, np.ndarray] = {}

        self._stimulus_files: Dict[str, h5py.File] = {}
        self.stimulus_path = str(stim_root / "S{}_stimuli_227.h5py")

        roi_list = list(selected_rois) if selected_rois is not None else None

        for subject in self.subject_id:
            self._load_subject(int(subject), roi_list, trn_image_idx)

        self.all_subjects = self.subject_id

    def _load_subject(self, subject: int, selected_rois, trn_image_idx) -> None:
        str_subject = str(subject)
        subj_image_idx = np.load(get_image_order_path(subject))
        subj_val_image_idx = np.load(Path(__file__).resolve().parent / "validation_idx.npy")

        if not trn_image_idx:
            subj_trn_image_idx = subj_image_idx[subj_image_idx >= 1000]
        elif isinstance(trn_image_idx, list):
            subj_trn_image_idx = trn_image_idx[self.subject_id.index(subject)]
        elif isinstance(trn_image_idx, str):
            subj_trn_image_idx = np.load(trn_image_idx)
        else:
            raise ValueError(f"Unsupported training_image_idx value: {type(trn_image_idx)}")

        self.trn_num_images[str_subject] = len(subj_trn_image_idx)
        self.val_num_images[str_subject] = len(subj_val_image_idx)
        self.trn_image_ids[str_subject] = np.asarray(subj_trn_image_idx)
        self.val_image_ids[str_subject] = np.asarray(subj_val_image_idx)

        if isinstance(selected_rois, str):
            selected_rois = [selected_rois]
        elif selected_rois is None:
            selected_rois = get_subject_rois(subject)

        train_reponse_trial_idx = [np.where(subj_image_idx == element)[0][0] for element in subj_trn_image_idx]
        val_reponse_trial_idx = [np.where(subj_image_idx == element)[0][0] for element in subj_val_image_idx]

        subject_neural_data = np.hstack([np.load(get_avg_response_path(subject, roi)) for roi in selected_rois])
        coordinate_data = np.vstack([np.load(get_coordinate_path(subject, roi)) for roi in selected_rois])
        #normalize the range of the coordinates
        coordinate_data[:, 0] = (coordinate_data[:, 0] / 182 + 92 / 182)
        coordinate_data[:, 1] = (coordinate_data[:, 1] / 218 + 126 / 218)
        coordinate_data[:, 2] = (coordinate_data[:, 2] / 182 + 72 / 182)
        self.mni_coordinates[str_subject] = coordinate_data

        self.trn_neural[str_subject] = subject_neural_data[train_reponse_trial_idx]
        self.val_neural[str_subject] = subject_neural_data[val_reponse_trial_idx]
        self.num_stimulus[str_subject] = len(subj_trn_image_idx)
        self.neural_sizes[str_subject] = subject_neural_data.shape[-1]

    def _get_stimulus_file(self, subject: int):
        subject_key = str(subject)
        if subject_key not in self._stimulus_files:
            self._stimulus_files[subject_key] = h5py.File(self.stimulus_path.format(subject), "r")
        return self._stimulus_files[subject_key]["stimuli"]

    def _load_image(self, subject: int, image_idx: int) -> np.ndarray:
        stimuli = self._get_stimulus_file(subject)
        return np.asarray(stimuli[int(image_idx)])

    def __len__(self):
        return max(self.num_stimulus.values())

    def __getitem__(self, idx):
        all_neural = []
        all_coordinates = []
        all_images = []

        for subject_idx in self.all_subjects:
            subject_key = str(subject_idx)
            curidx = idx if idx <= (self.num_stimulus[subject_key] - 1) else random.randint(0, self.num_stimulus[subject_key] - 1)

            selected_neural = self.trn_neural[subject_key][curidx]
            selected_coordinates = self.mni_coordinates[subject_key]

            if self.subsample_voxels and self.num_voxels is not None and self.num_voxels < selected_neural.shape[0]:
                sub_idx = np.random.choice(selected_neural.shape[0], self.num_voxels, replace=False)
                selected_neural = selected_neural[sub_idx]
                selected_coordinates = selected_coordinates[sub_idx, :]

            if self.shuffle_coord:
                perm = np.random.permutation(selected_neural.shape[0])
                selected_neural = selected_neural[perm]
                selected_coordinates = selected_coordinates[perm, :]

            all_neural.append(selected_neural)
            all_coordinates.append(selected_coordinates)

            image_id = self.trn_image_ids[subject_key][curidx]
            all_images.append(self._load_image(subject_idx, int(image_id)))

        batch = {
            "subject_id": torch.from_numpy(np.array([int(x) for x in self.all_subjects])),
            "mni_coordinate": torch.from_numpy(np.concatenate(all_coordinates)),
            "neural_data": torch.from_numpy(np.concatenate(all_neural)),
        }

        batch["image_data"] = torch.from_numpy(np.array(all_images))

        return batch

    def _get_split_item(self, subject_id, image_idx, split: str):
        subject_key = str(subject_id)
        batch = {
            "subject_id": subject_id,
            "mni_coordinate": torch.from_numpy(self.mni_coordinates[subject_key]),
        }

        if split == "train":
            neural_source = self.trn_neural[subject_key]
            image_ids = self.trn_image_ids[subject_key]
        elif split == "val":
            neural_source = self.val_neural[subject_key]
            image_ids = self.val_image_ids[subject_key]
        else:
            raise ValueError(f"Unsupported split: {split}")

        batch["neural_data"] = torch.from_numpy(neural_source[image_idx])
        images = [self._load_image(subject_id, int(img_id)) for img_id in image_ids[image_idx]]
        batch["image_data"] = torch.from_numpy(np.array(images))

        return batch

    def get_test_item(self, subject_id, image_idx):
        return self._get_split_item(subject_id, image_idx, split="val")

    def get_train_item(self, subject_id, image_idx):
        return self._get_split_item(subject_id, image_idx, split="train")

    def __del__(self):
        for file_handle in self._stimulus_files.values():
            try:
                file_handle.close()
            except Exception:
                pass
