from __future__ import annotations

from pathlib import Path

from train.nrf_trainer import NRFTrainer
from utils.config import SAVE_ROOT, arg_parser, load_experiment_config

SUPPORTED_MODES = {"train", "finetune", "evaluate"}
PROJECT_ROOT = Path(__file__).resolve().parent


def run(args) -> None:
    # Resolve experiment config path
    if args.pretrained_config_dir:
        config_path = Path(SAVE_ROOT) / args.pretrained_config_dir / "experiment_config.yaml"
    elif args.exp_config_dir:
        config_path = PROJECT_ROOT / args.exp_config_dir
    else:
        raise ValueError("Provide --exp_config_dir or --pretrained_config_dir.")

    config = load_experiment_config(config_path, args=args)

    # Auto-detect finetune mode
    if args.pretrained_config_dir and config["training"]["mode"] != "evaluate":
        config["training"]["mode"] = "finetune"

    # Default evaluation subjects and ROIs
    if not config["evaluation"].get("subject_list"):
        config["evaluation"]["subject_list"] = config["data_src"]["data_subject_list"]
    if not config["evaluation"].get("roi_list"):
        config["evaluation"]["roi_list"] = list(config["data_src"]["roi_list"])

    # For finetune, eval defaults to the finetune data subjects
    if config["training"]["mode"] == "finetune":
        config["evaluation"]["subject_list"] = (
            args.eval_subject_list if args.eval_subject_list else config["data_src"]["data_subject_list"]
        )

    mode = config["training"]["mode"]
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of {sorted(SUPPORTED_MODES)}.")

    # Setup experiment directory
    log_path = None
    if args.exp_name:
        log_path = Path(SAVE_ROOT) / args.exp_name
        log_path.mkdir(parents=True, exist_ok=True)

    trainer = NRFTrainer(str(log_path) if log_path else None, config, mode=mode)

    if mode in {"train", "finetune"}:
        trainer.train(config)
    else:
        trainer.evaluate(
            config["evaluation"]["subject_list"],
            eval_name=config["evaluation"].get("output_name") or "evaluation",
        )


if __name__ == "__main__":
    run(arg_parser())
