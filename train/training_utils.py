from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr


def gaussian_fourier_feature(b_matrix: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    def pe(x: torch.Tensor) -> torch.Tensor:
        return torch.concatenate([torch.sin(x @ b_matrix), torch.cos(x @ b_matrix)], axis=-1)

    return pe


def get_embed_fn(
    config: Dict,
    mode: str,
    log_path: str | Path | None,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gaussian_cfg = config.get("gaussian", {})
    std = gaussian_cfg["std"]
    num_features = gaussian_cfg["num_features"]
    b_matrix_path = gaussian_cfg.get("B_matrix_dir")

    if mode == "train" and not b_matrix_path:
        rng = np.random.default_rng(seed=42)
        b_matrix = rng.normal(scale=std, size=(3, num_features)).astype(np.float32)
        if log_path:
            b_matrix_path = os.path.join(str(log_path), "B.npy")
            np.save(b_matrix_path, b_matrix)
            config.setdefault("gaussian", {})["B_matrix_dir"] = b_matrix_path
    elif b_matrix_path:
        b_matrix = np.load(b_matrix_path)
    else:
        raise ValueError(
            "Gaussian B matrix dir not specified. "
            "For evaluate/finetune runs, point positional_embed.gaussian.B_matrix_dir to the saved B.npy "
            "from the training run, or load a saved experiment config."
        )

    b_matrix_t = torch.tensor(b_matrix, dtype=torch.float32, device=device)
    embed_fn = gaussian_fourier_feature(b_matrix_t)
    pe_dim = num_features * 2
    return embed_fn, pe_dim


def compute_loss(prediction: torch.Tensor, target: torch.Tensor, loss_type: str = "combine") -> torch.Tensor:
    prediction = torch.as_tensor(prediction)
    target = torch.as_tensor(target)

    if loss_type == "mse":
        return F.mse_loss(prediction, target)
    if loss_type == "combine":
        mse_loss = F.mse_loss(prediction.reshape(target.shape), target)
        cosine = torch.mean(F.cosine_similarity(prediction.reshape(target.shape), target))
        return 0.9 * mse_loss - 0.1 * cosine
    raise ValueError(f"Unsupported loss type: {loss_type}")


def compute_scores(prediction, target) -> np.ndarray:
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    corrs = []
    for voxel_idx in range(prediction.shape[-1]):
        corr = pearsonr(np.squeeze(prediction[:, voxel_idx]), np.squeeze(target[:, voxel_idx]))[0]
        corrs.append(corr if not np.isnan(corr) else 0.0)
    return np.array(corrs)


def save_checkpoint(path: Path | str, state: dict) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)


def load_checkpoint(path: Path | str, device: torch.device) -> Optional[dict]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None
    return torch.load(checkpoint_path, map_location=device)
