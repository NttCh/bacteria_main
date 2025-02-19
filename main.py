#!/usr/bin/env python
"""
Main application for training, evaluating, and exporting the bacteria classification model.

This script loads configuration from a YAML file (or an internal dictionary), sets up the training pipeline
using the bacteria_lib, and then runs the training process.
"""

import sys
import os

# (Optional) Debug snippet: ensure we see the correct environment paths
print("DEBUG: CWD =", os.getcwd())
print("DEBUG: sys.path BEFORE =", sys.path)

# If needed, forcibly add the library path:
# LIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bacteria_lib"))
# if LIB_PATH not in sys.path:
#     sys.path.insert(0, LIB_PATH)

print("DEBUG: sys.path AFTER =", sys.path)

import datetime
import pandas as pd
import torch
import optuna
from omegaconf import OmegaConf, DictConfig

# Import from your library
from bacteria_lib.train_pipeline import (
    train_stage,
    train_with_cross_validation,
    continue_training,
    evaluate_model,
    evaluate_on_test,
    LitClassifier
)
from bacteria_lib.callbacks import PlotMetricsCallback, OptunaReportingCallback
from bacteria_lib.utils import set_seed
from bacteria_lib import __version__ as lib_version

def main() -> None:
    """
    Main function to run training, evaluation, and model export.
    """
    print(f"Using bacteria_lib version: {lib_version}")

    # 1. Load configuration
    cfg_path = "config.yaml"
    if os.path.exists(cfg_path):
        cfg = OmegaConf.load(cfg_path)
    else:
        # fallback to an internal dictionary if you prefer
        raise FileNotFoundError(f"No config.yaml found at {cfg_path}")

    # 2. Set random seed
    set_seed(cfg.training.seed)

    # 3. Create output directories
    date_folder = datetime.datetime.now().strftime("%Y%m%d")
    base_save_dir = os.path.join(cfg.general.save_dir, date_folder)
    os.makedirs(base_save_dir, exist_ok=True)
    best_model_folder = os.path.join(base_save_dir, "best_model")
    os.makedirs(best_model_folder, exist_ok=True)

    # 4. Initialize a list to store evaluation metrics
    metrics_data = []

    # 5. Stage 1: Detection
    if cfg.stage_detection:
        # same logic as your original detection stage...
        pass

    # 6. Stage 2: Classification
    if cfg.stage_classification:
        print("=== Stage 2: Classification (multi-class) ===")
        # If Optuna is used, do the hyperparameter search
        if cfg.optuna.use_optuna:
            pass  # same logic to run study, etc.

        # Check cross_validation or single-split
        classification_csv = cfg.data.classification_csv
        if cfg.training.cross_validation:
            classification_model, classification_cv_acc = train_with_cross_validation(
                cfg, classification_csv, num_classes=9, stage_name="classification"
            )
            print(f"Classification (CV) Average Accuracy: {classification_cv_acc:.4f}")
            metrics_data.append({"stage": "classification_CV", "val_acc": float(classification_cv_acc)})
        else:
            classification_model, classification_val_acc = train_stage(
                cfg, classification_csv, num_classes=9, stage_name="classification"
            )
            print(f"Classification Stage Validation Accuracy: {classification_val_acc:.4f}")
            metrics_data.append({"stage": "classification", "val_acc": float(classification_val_acc.item())})

        # Save the best classification model checkpoint
        classification_checkpoint = os.path.join(best_model_folder, "best_classification.ckpt")
        torch.save(classification_model.state_dict(), classification_checkpoint)
        print(f"Saved classification best model checkpoint to {classification_checkpoint}")

        # Continue training
        classification_model = continue_training(
            classification_model, cfg, classification_csv, num_classes=9, stage_name="classification"
        )
        evaluate_model(classification_model, classification_csv, cfg, stage="Classification")

    # 7. Save Evaluation Metrics
    import pandas as pd
    metrics_df = pd.DataFrame(metrics_data)
    eval_table_path = os.path.join(base_save_dir, "evaluation_metrics.csv")
    metrics_df.to_csv(eval_table_path, index=False)
    print(f"Saved evaluation metrics to {eval_table_path}")

    # 8. Export Model
    if cfg.stage_classification:
        classification_model.load_state_dict(torch.load(classification_checkpoint))
        classification_model.eval()
        scripted_model = torch.jit.script(classification_model)
        export_path = os.path.join(best_model_folder, "best_classification_scripted.pt")
        scripted_model.save(export_path)
        print(f"Exported scripted classification model to {export_path}")

    # 9. Optional: Evaluate on a dedicated test set
    if "test_csv" in cfg.data and os.path.exists(cfg.data.test_csv):
        print("Evaluating on the dedicated test set...")
        evaluate_on_test(classification_model, cfg.data.test_csv, cfg)

    print("Training finished. Best models and evaluation metrics are saved in:", best_model_folder)

if __name__ == "__main__":
    main()
