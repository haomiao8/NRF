"""Project configuration and experiment config loading.

Project config (config.yaml): paths and per-subject ROI lists.
Experiment config (per-run YAML): training, model, data, evaluation settings.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

# ── YAML I/O ──

def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file.  If *path* is a directory, loads experiment_config.yaml inside it."""
    p = Path(path)
    if p.is_dir():
        p = p / "experiment_config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {p} must be a YAML mapping.")
    return data


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ── Project-level config (paths + per-subject ROIs) ──

def _load_project_config() -> Dict[str, Any]:
    return load_yaml(Path(__file__).resolve().parent.parent / "config.yaml")


_PROJECT = _load_project_config()
PATHS = _PROJECT["paths"]
SUBJECT_ROIS = _PROJECT["subject_rois"]

DATA_ROOT = PATHS["data_root"]
SAVE_ROOT = PATHS["save_root"]


# ── Path helpers ──



def get_avg_response_dir(subject: int | str) -> Path:
    return Path(DATA_ROOT) / PATHS["avg_response_dirname"] / f"subj{subject:02d}"


def get_image_order_path(subject: int | str) -> Path:
    return get_avg_response_dir(subject) / PATHS["image_order_filename"]


def get_avg_response_path(subject: int | str, roi: str) -> Path:
    return get_avg_response_dir(subject) / f"{roi}.npy"


def get_coordinate_path(subject: int | str, roi: str) -> Path:
    return Path(DATA_ROOT) / PATHS["coordinate_dirname"] / f"subj{subject:02d}" / f"{roi}_MNI_coordinate.npy"


# ── Experiment config: load → defaults → CLI overrides → validate ──

def _apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing keys with sensible defaults."""
    config.setdefault("training", {})
    config.setdefault("positional_embed", {})
    config.setdefault("data_src", {})
    config.setdefault("model", {})
    config.setdefault("finetune", {})
    config.setdefault("evaluation", {})
    config.setdefault("feature_merger", {})
    config.setdefault("image_embedding", {})

    config["data_src"]["dataset"] = "nsd"

    pe = config["positional_embed"]
    pe["type"] = "gaussian"
    g = pe.setdefault("gaussian", {})
    g.setdefault("B_matrix_dir", None)
    g.setdefault("std", 32)
    g.setdefault("num_features", 20)

    t = config["training"]
    t.setdefault("mode", "train")
    t.setdefault("shuffle_coord", False)
    t.setdefault("subsample_voxels", False)
    t.setdefault("num_voxels", None)
    t.setdefault("num_workers", 0)
    t.setdefault("model_save_iter", None)

    ft = config["finetune"]
    ft.setdefault("pretrained_config_dir", None)
    ft.setdefault("pretrained_ckpt_name", None)
    ft.setdefault("train_encoder", False)
    ft.setdefault("train_feature_merger", False)

    ev = config["evaluation"]
    ev.setdefault("subject_list", None)
    ev.setdefault("roi_list", None)
    ev.setdefault("data_type", None)
    ev.setdefault("output_name", None)

    config["image_embedding"].setdefault("model_name", "openai/clip-vit-base-patch16")
    config["image_embedding"].setdefault("selected_layers", [3, 6])

    return config


def _apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Override config values with explicitly-provided CLI arguments.

    Each CLI argument maps to exactly one config key — no recursive matching.
    """
    def _set(section: str, key: str, cli_attr: str | None = None):
        val = getattr(args, cli_attr or key, None)
        if val is not None:
            config[section][key] = val

    # Training
    _set("training", "mode")
    _set("training", "epochs")
    _set("training", "batch_size")
    _set("training", "learning_rate")
    _set("training", "shuffle_coord")
    _set("training", "subsample_voxels")
    _set("training", "num_voxels")
    _set("training", "model_save_iter")

    # Data source
    _set("data_src", "data_subject_list")
    _set("data_src", "roi_list")
    _set("data_src", "training_image_idx")
    _set("data_src", "model_subject_list")

    # Model architecture
    _set("model", "num_layers")
    _set("model", "hidden_layer_dim")
    _set("model", "latent_in")
    _set("model", "dropout_layer")
    _set("model", "dropout_prob")
    _set("model", "weight_norm")
    _set("model", "weight_norm_layer")

    # Positional embedding
    if getattr(args, "num_features", None) is not None:
        config["positional_embed"]["gaussian"]["num_features"] = args.num_features
    if getattr(args, "std", None) is not None:
        config["positional_embed"]["gaussian"]["std"] = args.std

    # Fine-tuning
    _set("finetune", "pretrained_config_dir")
    _set("finetune", "pretrained_ckpt_name")
    _set("finetune", "train_encoder")
    _set("finetune", "train_feature_merger")

    # Evaluation
    _set("evaluation", "subject_list", "eval_subject_list")
    _set("evaluation", "roi_list", "eval_roi_list")
    _set("evaluation", "data_type", "eval_data_type")
    _set("evaluation", "output_name", "eval_output_name")


def _validate(config: Dict[str, Any]) -> None:
    if not config["data_src"].get("data_subject_list"):
        raise ValueError("data_src.data_subject_list must be provided.")
    mode = config["training"].get("mode", "train")
    if mode in {"train", "finetune"}:
        for key in ("epochs", "batch_size", "learning_rate"):
            if key not in config["training"]:
                raise ValueError(f"training.{key} must be specified for mode='{mode}'.")


def load_experiment_config(path: str | Path, args: argparse.Namespace | None = None) -> Dict[str, Any]:
    """Load experiment YAML → apply defaults → apply CLI overrides → validate."""
    config = _apply_defaults(copy.deepcopy(load_yaml(path)))
    if args is not None:
        _apply_cli_overrides(config, args)
    _validate(config)
    return config


# ── CLI argument parser ──

def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NRF training / fine-tuning / evaluation")

    # Experiment
    parser.add_argument("--exp_name", type=str, help="Experiment name (output directory under save_root)")
    parser.add_argument("--exp_config_dir", type=str, help="Path to experiment config YAML (file or directory)")
    parser.add_argument("--mode", type=str, default=None, help="train / finetune / evaluate")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--model_save_iter", type=int)
    parser.add_argument("--shuffle_coord", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--subsample_voxels", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num_voxels", type=int)

    # Data
    parser.add_argument("--data_subject_list", nargs="+", help="Training subject id(s)")
    parser.add_argument("--model_subject_list", nargs="+")
    parser.add_argument("--roi_list", nargs="+", type=str)
    parser.add_argument("--training_image_idx", type=str, help="Path to .npy with training image ids")

    # Model architecture (usually set in config YAML, not CLI)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--hidden_layer_dim", type=int)
    parser.add_argument("--latent_in", nargs="+", type=int)
    parser.add_argument("--dropout_layer", type=float)
    parser.add_argument("--dropout_prob", type=float)
    parser.add_argument("--weight_norm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--weight_norm_layer", nargs="+", type=int)
    parser.add_argument("--num_features", type=int, help="Gaussian Fourier feature dimension")
    parser.add_argument("--std", type=int, help="Std of Gaussian B matrix")

    # Fine-tuning
    parser.add_argument("--pretrained_config_dir", type=str, help="Previous experiment dir under save_root")
    parser.add_argument("--pretrained_ckpt_name", type=str, help="Checkpoint name (e.g. best_model)")
    parser.add_argument("--train_encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--train_feature_merger", action=argparse.BooleanOptionalAction, default=None)

    # Evaluation
    parser.add_argument("--eval_subject_list", nargs="+")
    parser.add_argument("--eval_roi_list", nargs="+", type=str)
    parser.add_argument("--eval_data_type", type=str)
    parser.add_argument("--eval_output_name", type=str)

    return parser.parse_args()
