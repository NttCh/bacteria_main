#!/usr/bin/env python
"""
Main application for training, evaluating, and exporting the bacteria classification model.

This script loads configuration from a YAML file, sets up the training pipeline
(using bacteria_lib), and then runs the training process.
"""

import os
import datetime
import torch
import pytorch_lightning as pl
import optuna
from omegaconf import OmegaConf, DictConfig

# Import functions and classes from the bacteria_lib library.
from bacteria_lib.data import PatchClassificationDataset
from bacteria_lib.models import build_classifier
from bacteria_lib.callbacks import PlotMetricsCallback, OptunaReportingCallback
from bacteria_lib.utils import set_seed
from bacteria_lib.train_pipeline import (
    train_stage,
    train_with_cross_validation,
    continue_training,
    evaluate_model,
    evaluate_on_test
)

def main() -> None:
    """
    Main function to run the training, evaluation, and model export pipeline.
    """
    # Load configuration from config.yaml.
    cfg_path = "config.yaml"
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    cfg: DictConfig = OmegaConf.load(cfg_path)

    # Set the random seed for reproducibility.
    set_seed(cfg.training.seed)

    # Create directories for logs and model checkpoints.
    date_folder = datetime.datetime.now().strftime("%Y%m%d")
    base_save_dir = os.path.join(cfg.general.save_dir, date_folder)
    os.makedirs(base_save_dir, exist_ok=True)
    best_model_folder = os.path.join(base_save_dir, "best_model")
    os.makedirs(best_model_folder, exist_ok=True)

    # List to store evaluation metrics.
    metrics_data = []

    # ---------- Stage 1: Detection (if enabled) ----------
    if cfg.stage_detection:
        print("=== Stage 1: Detection (binary classification) ===")
        if cfg.optuna.use_optuna:
            study_det = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2)
            )
            study_det.optimize(
                lambda trial: train_stage(cfg, cfg.data.detection_csv, num_classes=2, stage_name="detection", trial=trial, suppress_metrics=True)[1],
                n_trials=cfg.optuna.n_trials
            )
            print("Best detection trial:", study_det.best_trial)
        # Use single-split training (or cross-validation if enabled)
        if cfg.training.cross_validation:
            detection_model, detection_cv_acc = train_with_cross_validation(cfg, cfg.data.detection_csv, num_classes=2, stage_name="detection")
            print(f"Detection (CV) Average Accuracy: {detection_cv_acc:.4f}")
            metrics_data.append({"stage": "detection_CV", "val_acc": float(detection_cv_acc)})
        else:
            detection_model, detection_val_acc = train_stage(cfg, cfg.data.detection_csv, num_classes=2, stage_name="detection")
            print(f"Detection Stage Validation Accuracy: {detection_val_acc:.4f}")
            metrics_data.append({"stage": "detection", "val_acc": float(detection_val_acc.item())})
        detection_checkpoint = os.path.join(best_model_folder, "best_detection.ckpt")
        torch.save(detection_model.state_dict(), detection_checkpoint)
        print(f"Saved detection best model checkpoint to {detection_checkpoint}")
        detection_model = continue_training(detection_model, cfg, cfg.data.detection_csv, num_classes=2, stage_name="detection")
        evaluate_model(detection_model, cfg.data.detection_csv, cfg, stage="Detection")

    # ---------- Stage 2: Classification ----------
    if cfg.stage_classification:
        print("=== Stage 2: Classification (multi-class) ===")
        if cfg.optuna.use_optuna:
            study_cls = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2)
            )
            study_cls.optimize(
                lambda trial: train_stage(cfg, cfg.data.classification_csv, num_classes=9, stage_name="classification", trial=trial, suppress_metrics=True)[1],
                n_trials=cfg.optuna.n_trials
            )
            print("Best classification trial:", study_cls.best_trial)
        if cfg.training.cross_validation:
            classification_model, classification_cv_acc = train_with_cross_validation(cfg, cfg.data.classification_csv, num_classes=9, stage_name="classification")
            print(f"Classification (CV) Average Accuracy: {classification_cv_acc:.4f}")
            metrics_data.append({"stage": "classification_CV", "val_acc": float(classification_cv_acc)})
        else:
            classification_model, classification_val_acc = train_stage(cfg, cfg.data.classification_csv, num_classes=9, stage_name="classification")
            print(f"Classification Stage Validation Accuracy: {classification_val_acc:.4f}")
            metrics_data.append({"stage": "classification", "val_acc": float(classification_val_acc.item())})
        classification_checkpoint = os.path.join(best_model_folder, "best_classification.ckpt")
        torch.save(classification_model.state_dict(), classification_checkpoint)
        print(f"Saved classification best model checkpoint to {classification_checkpoint}")
        classification_model = continue_training(classification_model, cfg, cfg.data.classification_csv, num_classes=9, stage_name="classification")
        evaluate_model(classification_model, cfg.data.classification_csv, cfg, stage="Classification")

    # ---------- Save Evaluation Metrics ----------
    metrics_df = pd.DataFrame(metrics_data)
    eval_table_path = os.path.join(base_save_dir, "evaluation_metrics.csv")
    metrics_df.to_csv(eval_table_path, index=False)
    print(f"Saved evaluation metrics to {eval_table_path}")

    # ---------- Model Export ----------
    if cfg.stage_classification:
        classification_model.load_state_dict(torch.load(classification_checkpoint))
        classification_model.eval()
        scripted_model = torch.jit.script(classification_model)
        export_path = os.path.join(best_model_folder, "best_classification_scripted.pt")
        scripted_model.save(export_path)
        print(f"Exported scripted classification model to {export_path}")

    # ---------- Optional: Test Set Evaluation ----------
    if "test_csv" in cfg.data and os.path.exists(cfg.data.test_csv):
        print("Evaluating on the dedicated test set...")
        evaluate_on_test(classification_model, cfg.data.test_csv, cfg)

    print("Training finished. Best models and evaluation metrics are saved in:", best_model_folder)


if __name__ == "__main__":
    main()
