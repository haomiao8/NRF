from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency in some clusters
    SummaryWriter = None
from tqdm import tqdm

from model.Feature_merger import Feature_merger
from model.image_encoder import CLIPImageEncoder
from model.nrf_encoder import NRF_Encoder
from train.dataloader import neural_loader
from utils.config import PATHS
from utils.config import save_config
from train.training_utils import compute_loss, compute_scores, get_embed_fn, load_checkpoint, save_checkpoint


def get_validation_image_count(dataset: Any, subject_id: Any, default_count: int = 907) -> int:
    if hasattr(dataset, "val_num_images"):
        value = dataset.val_num_images.get(str(subject_id))
        if value is not None:
            return int(value)
    return default_count


class NRFTrainer:
    def __init__(
        self,
        log_path: Optional[str],
        config: Dict[str, Any],
        mode: str,
    ):
        resolved_mode = config.get("training", {}).get("mode", mode)

        self.config = config
        self.mode = resolved_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_path = Path(log_path) if log_path else None
        self.logger_tf = SummaryWriter(log_dir=self.log_path / "log") if (self.log_path and SummaryWriter is not None) else None

        training_cfg = config["training"]
        finetune_cfg = config.get("finetune", {})

        self.epochs = training_cfg["epochs"]
        self.patience = training_cfg.get("patience", 5)
        self.shuffle_coord = bool(training_cfg.get("shuffle_coord", False))
        self.subsample_voxels = bool(training_cfg.get("subsample_voxels", False))
        self.num_voxels = training_cfg.get("num_voxels", None)
        self.train_encoder = bool(finetune_cfg.get("train_encoder", False))
        self.train_feature_merger = bool(finetune_cfg.get("train_feature_merger", False))

        self.iteration = 0
        self.cur_epoch = 0
        self.best_val_corr = -1.0
        self.best_epoch = -1
        self.early_stop_counter = 0

        self.model = None
        self.feature_merger_model = None
        self.image_encoder_model = None
        self.optim = None
        self.scheduler = None
        self.dataset = None
        self.trn_dataloader = None

        self._setup_embeddings()
        self._setup_image_encoder()
        self._setup_encoder()
        self._set_training_flags()
        self._set_model_mode()

        if self.mode in ("finetune", "evaluate"):
            self.load_weights_and_checkpoint()

        if self.mode in ("train", "finetune"):
            self._setup_training()
        elif self.mode == "evaluate":
            self._setup_evaluation_dataset()

        if self.log_path is not None:
            save_config(config, self.log_path / "experiment_config.yaml")
            self._save_run_metadata()

    def _save_run_metadata(self):
        if self.log_path is None:
            return
        training_cfg = self.config.get("training", {})
        metadata = {
            "seed": training_cfg.get("seed"),
            "mode": self.mode,
            "data_subject_list": self.config.get("data_src", {}).get("data_subject_list", []),
            "image_encoder": self.config.get("image_embedding", {}).get("model_name", "openai/clip-vit-base-patch16"),
        }
        with open(self.log_path / "run_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    def _setup_embeddings(self):
        self.embed_fn, self.pe_dim = get_embed_fn(
            self.config["positional_embed"],
            mode=self.mode,
            log_path=self.log_path,
        )

    def _setup_image_encoder(self):
        image_cfg = self.config.get("image_embedding", {})
        self.image_encoder_model = CLIPImageEncoder(
            model_name=image_cfg.get("model_name", "openai/clip-vit-base-patch16"),
            selected_layers=image_cfg.get("selected_layers", [3, 6]),
        ).to(self.device)

    def loss(self, prediction: torch.Tensor, targets: torch.Tensor, loss_type: str) -> torch.Tensor:
        return compute_loss(prediction, targets, loss_type)

    def scores(self, prediction, targets) -> np.ndarray:
        return compute_scores(prediction, targets)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        if self.logger_tf is not None:
            self.logger_tf.add_scalar(name, value, step)

    def _setup_encoder(self):
        feature_cfg = self.config.get("feature_merger", {})
        self.image_embedding_dim = feature_cfg.get("higher_feat_input_dim", 512)
        self.feature_merger_model = Feature_merger(
            feature_cfg["early_feat_input_dim"],
            feature_cfg["num_patches"],
            feature_cfg["downsample_dim"],
        ).to(self.device)
        self.image_embedding_dim = (2 * feature_cfg["downsample_dim"]) + feature_cfg["higher_feat_input_dim"]

        self.model = NRF_Encoder(
            self.image_embedding_dim,
            self.pe_dim,
            **self.config["model"],
        ).to(self.device)

    def _set_training_flags(self):
        if self.mode == "train":
            self.train_encoder = True
            self.train_feature_merger = True

    def _set_model_mode(self):
        self.model.train() if self.train_encoder else self.model.eval()
        if self.feature_merger_model is not None:
            if self.train_feature_merger and self.mode in ("train", "finetune"):
                self.feature_merger_model.train()
            else:
                self.feature_merger_model.eval()
        if self.image_encoder_model is not None:
            if any(param.requires_grad for param in self.image_encoder_model.parameters()) and self.mode in ("train", "finetune"):
                self.image_encoder_model.train()
            else:
                self.image_encoder_model.eval()

    def _setup_training(self):
        all_params = []
        if self.train_encoder:
            all_params.extend(self.model.parameters())
        if self.train_feature_merger and self.feature_merger_model is not None:
            all_params.extend(self.feature_merger_model.parameters())
        self.optim = torch.optim.AdamW(
            all_params,
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["wdecay"],
        ) if all_params else None

        self.dataset = self._build_dataset(
            subject_id=self.config["data_src"]["data_subject_list"],
            selected_rois=self.config["data_src"]["roi_list"],
            selected_layers=self.config["image_embedding"].get("selected_layers", [3, 6]),
            shuffle_coord=self.shuffle_coord,
            subsample_voxels=self.subsample_voxels,
            trn_image_idx=self.config["data_src"].get("training_image_idx", None),
            num_voxels=self.num_voxels,
        )
        self.trn_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["training"].get("num_workers", 4),
        )

    def _build_dataset(
        self,
        subject_id,
        selected_rois,
        selected_layers,
        shuffle_coord: bool,
        trn_image_idx,
        subsample_voxels: bool = False,
        num_voxels: Optional[int] = None,
    ):
        return neural_loader(
            subject_id=subject_id,
            selected_rois=selected_rois,
            selected_layers=selected_layers,
            shuffle_coord=shuffle_coord,
            subsample_voxels=subsample_voxels,
            trn_image_idx=trn_image_idx,
            num_voxels=num_voxels,
        )

    def _setup_evaluation_dataset(self):
        eval_subjects = self._resolve_eval_subjects()
        eval_rois = self._resolve_eval_rois()
        self.dataset = self._build_dataset(
            subject_id=eval_subjects,
            selected_rois=eval_rois,
            selected_layers=self.config["image_embedding"].get("selected_layers", [3, 6]),
            shuffle_coord=False,
            trn_image_idx=self.config["data_src"].get("training_image_idx", None),
        )

    def _resolve_eval_subjects(self):
        evaluation_cfg = self.config.get("evaluation", {})
        return evaluation_cfg.get("subject_list") or self.config["data_src"]["data_subject_list"]

    def _resolve_eval_rois(self):
        evaluation_cfg = self.config.get("evaluation", {})
        return evaluation_cfg.get("roi_list") or self.config["data_src"]["roi_list"]

    def load_weights_and_checkpoint(self):
        ft = self.config.get("finetune", {})
        pretrained_cfg_dir = ft.get("pretrained_config_dir")
        pretrained_ckpt_name = ft.get("pretrained_ckpt_name")
        if not pretrained_cfg_dir or not pretrained_ckpt_name:
            return

        pretrained_path = Path(PATHS["save_root"]) / pretrained_cfg_dir / pretrained_ckpt_name
        model_ckpt = load_checkpoint(pretrained_path / "best_NRF_model.pth.tar", self.device)
        if model_ckpt is not None and self.model is not None:
            self.model.load_state_dict(model_ckpt["state_dict"])

        if self.feature_merger_model is not None:
            merger_ckpt = load_checkpoint(pretrained_path / "best_feature_merger.pth.tar", self.device)
            if merger_ckpt is not None:
                self.feature_merger_model.load_state_dict(merger_ckpt["state_dict"])

    def prepare_batch_neural(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        subject = batch["subject_id"]
        neural_data = batch["neural_data"].to(self.device)
        coordinate_data = batch["mni_coordinate"].to(torch.float32).to(self.device)
        early_image_data = batch.get("early_image_data")
        higher_image_data = batch.get("higher_image_data")
        image_data = batch.get("image_data")

        if early_image_data is not None:
            early_image_data = early_image_data.to(self.device)
        if higher_image_data is not None:
            higher_image_data = higher_image_data.to(self.device)
        if image_data is not None:
            image_data = image_data.to(self.device)
            if image_data.ndim == 5:
                if image_data.shape[1] != 1:
                    raise ValueError("Online image extraction currently supports single-subject batches only.")
                image_data = image_data.squeeze(1)

        return subject, early_image_data, higher_image_data, image_data, coordinate_data, neural_data

    def _encode_images(self, image_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.image_encoder_model is None:
            raise RuntimeError("No image encoder has been initialized.")
        return self.image_encoder_model(image_data)

    def run_model(
        self,
        early_embed_vector: Optional[torch.Tensor],
        higher_embed_vector: Optional[torch.Tensor],
        coordinates: torch.Tensor,
        image_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if image_data is not None:
            early_embed_vector, higher_embed_vector = self._encode_images(image_data)
        if early_embed_vector is None or higher_embed_vector is None:
            raise ValueError("Missing image features for model forward pass.")

        pe_coords = self.embed_fn(coordinates.to(torch.float).to(self.device))
        num_coords = pe_coords.shape[1]
        num_images = higher_embed_vector.shape[0]

        merged_early = self.feature_merger_model([early_embed_vector[:, 0], early_embed_vector[:, 1]])
        final_embed = torch.concatenate([merged_early, higher_embed_vector], dim=-1)
        final_embed = final_embed.unsqueeze(1).repeat(1, num_coords, 1)
        inputs = torch.cat((final_embed, pe_coords), dim=-1)
        if inputs.shape[0] != num_images * num_coords:
            inputs = inputs.reshape(num_images * num_coords, -1)
        prediction = self.model(inputs)
        return prediction.reshape(num_images, num_coords)

    def train_step(self, prediction: torch.Tensor, targets: torch.Tensor, loss_type: str) -> float:
        loss = self.loss(prediction, targets, loss_type)
        if self.optim is not None:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return float(loss.detach().cpu().numpy())

    def evaluate(self, eval_subject_list, eval_name: Optional[str] = None, split: str = "val"):
        if self.dataset is None:
            self._setup_evaluation_dataset()
        all_correlations = []
        eval_data = {}
        with torch.no_grad():
            for subject_id in eval_subject_list:
                if split == "train":
                    num_images = int(self.dataset.trn_num_images[str(subject_id)])
                    get_item = self.dataset.get_train_item
                elif split == "val":
                    num_images = get_validation_image_count(self.dataset, subject_id, default_count=907)
                    get_item = self.dataset.get_test_item
                else:
                    raise ValueError(f"Unsupported evaluation split: {split}")
                subject_predictions = []
                subject_targets = []
                for idx in tqdm(range(0, num_images), leave=False):
                    image_idx = np.arange(idx, min(idx + 1, num_images))
                    test_data = get_item(subject_id, image_idx)
                    test_data["mni_coordinate"] = test_data["mni_coordinate"].unsqueeze(0).repeat(len(image_idx), 1, 1)
                    _, early, higher, image_data, coords, targets = self.prepare_batch_neural(test_data)
                    predictions = self.run_model(early, higher, coords, image_data=image_data)
                    subject_predictions.append(predictions.cpu().numpy().astype(np.float32))
                    subject_targets.append(targets.cpu().numpy().astype(np.float32))

                pred_np = np.concatenate(subject_predictions, axis=0)
                tgt_np = np.concatenate(subject_targets, axis=0)
                corr = self.scores(pred_np, tgt_np)
                eval_data[subject_id] = {
                    "predictions": pred_np,
                    "targets": tgt_np,
                    "correlations": corr,
                }
                all_correlations.extend(corr)

        if eval_name is not None:
            self.save_evaluation_results(eval_data, eval_name)
        return (np.median(all_correlations) if all_correlations else -1.0), eval_data

    def save_evaluation_results(self, eval_data, eval_name: str):
        if self.log_path is None:
            return
        out_path = self.log_path / f"{eval_name}.h5py"
        with h5py.File(out_path, "w", libver="latest") as file_handle:
            for subject_id, data in eval_data.items():
                file_handle.create_dataset(f"subj{subject_id}_gt", data=data["targets"], dtype=np.float32)
                file_handle.create_dataset(f"subj{subject_id}_pred", data=data["predictions"], dtype=np.float32)
                file_handle.create_dataset(f"subj{subject_id}_corr", data=data["correlations"], dtype=np.float32)

    def _save_best_models(self, epoch: int, val_corr: float):
        if self.log_path is None:
            return
        best_dir = self.log_path / "best_model"
        save_checkpoint(
            best_dir / "best_NRF_model.pth.tar",
            {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optim_dict": self.optim.state_dict() if self.optim else None,
                "val_corr": val_corr,
            },
        )
        if self.feature_merger_model is not None:
            save_checkpoint(
                best_dir / "best_feature_merger.pth.tar",
                {
                    "epoch": epoch,
                    "state_dict": self.feature_merger_model.state_dict(),
                    "optim_dict": self.optim.state_dict() if self.optim else None,
                    "val_corr": val_corr,
                },
            )
    def train(self, config: Dict[str, Any]):
        loss_type = config["training"].get("loss_type", "combine")
        best_eval_data = None
        eval_subject_list = self._resolve_eval_subjects()

        for epoch in range(self.epochs):
            self.cur_epoch = epoch
            self._set_model_mode()
            epoch_losses = []
            print(f"[Epoch {epoch + 1}/{self.epochs}] Training...")

            for batch in self.trn_dataloader:
                _, early, higher, image_data, coordinates, targets = self.prepare_batch_neural(batch)
                prediction = self.run_model(early, higher, coordinates, image_data=image_data)
                trn_loss = self.train_step(prediction, targets, loss_type)
                epoch_losses.append(trn_loss)
                if self.iteration % 100 == 0:
                    self.log_scalar("trn_loss", trn_loss, self.iteration)
                self.iteration += 1

            avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            self.log_scalar("epoch_avg_loss", avg_epoch_loss, epoch)
            self.model.eval()
            if self.feature_merger_model is not None:
                self.feature_merger_model.eval()
            if self.image_encoder_model is not None:
                self.image_encoder_model.eval()

            val_corr, eval_data = self.evaluate(eval_subject_list, str(epoch), split="val")
            self.log_scalar("val_correlation", float(val_corr), epoch)
            print(
                f"[Epoch {epoch + 1}/{self.epochs}] "
                f"avg_train_loss={avg_epoch_loss:.6f}, val_corr={float(val_corr):.6f}"
            )
            if val_corr > self.best_val_corr:
                self.best_val_corr = float(val_corr)
                self.best_epoch = epoch
                self.early_stop_counter = 0
                self._save_best_models(epoch, float(val_corr))
                best_eval_data = eval_data
                if self.mode == "finetune":
                    self.evaluate(eval_subject_list, eval_name="ft_data", split="train")
                print(f"[Epoch {epoch + 1}/{self.epochs}] New best checkpoint saved.")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print(
                        f"[Epoch {epoch + 1}/{self.epochs}] Early stopping "
                        f"(patience={self.patience})"
                    )
                    break

        if best_eval_data is not None:
            self.save_evaluation_results(best_eval_data, "best_epoch")
