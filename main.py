#!/usr/bin/env python
"""
Main application for training, evaluating, and exporting the bacteria classification model.
"""

import os
import datetime
import pandas as pd
import pytorch_lightning as pl
import optuna

from omegaconf import OmegaConf, DictConfig

# Import from the library
from bacteria_lib.data import PatchClassificationDataset
from bacteria_lib.models import build_classifier
from bacteria_lib.callbacks import PlotMetricsCallback, OptunaReportingCallback
from bacteria_lib.utils import load_obj, set_seed

# (Optionally, import transforms if needed)
# from bacteria_lib.transforms import ToGray3

# Configuration (could also be loaded from a config.yaml file)
CONFIG_DICT = {
    "stage_detection": False,
    "stage_classification": True,
    "general": {
        "save_dir": "baclog_n1",
        "project_name": "bacteria_n1"
    },
    "trainer": {
        "devices": 1,
        "accelerator": "auto",
        "precision": "16-mixed",
        "log_every_n_steps": 10
    },
    "training": {
        "seed": 666,
        "mode": "max",
        "tuning_epochs_detection": 5,
        "tuning_epochs_classification": 5,
        "additional_epochs_detection": 5,
        "additional_epochs_classification": 5,
        "cross_validation": False,
        "num_folds": 3
    },
    "optimizer": {
        "class_name": "torch.optim.AdamW",
        "params": {"lr": 1e-4, "weight_decay": 0.001}
    },
    "scheduler": {
        "class_name": "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "step": "epoch",
        "monitor": "val_acc",
        "params": {"mode": "max", "factor": 0.1, "patience": 3}
    },
    "model": {
        "backbone": {
            "class_name": "torchvision.models.resnet50",
            "params": {"weights": "ResNet50_Weights.IMAGENET1K_V1"}
        }
    },
    "data": {
        "detection_csv": r"path/to/stage1_train.csv",
        "classification_csv": r"path/to/stage2_train.csv",
        "test_csv": r"path/to/test.csv",
        "folder_path": r"path/to/images",
        "num_workers": 0,
        "batch_size": 4,
        "label_col": "label",
        "valid_split": 0.2
    },
    "augmentation": {
        "train": {
            "augs": [
                {"class_name": "albumentations.Resize", "params": {"height": 400, "width": 400, "p": 1.0}},
                {"class_name": "__main__.ToGray3", "params": {"p": 1.0}},
                {"class_name": "albumentations.HorizontalFlip", "params": {"p": 0.5}},
                {"class_name": "albumentations.RandomBrightnessContrast", "params": {"p": 0.5}},
                {"class_name": "albumentations.Normalize", "params": {}},
                {"class_name": "albumentations.pytorch.transforms.ToTensorV2", "params": {"p": 1.0}}
            ]
        },
        "valid": {
            "augs": [
                {"class_name": "albumentations.Resize", "params": {"height": 400, "width": 400, "p": 1.0}},
                {"class_name": "__main__.ToGray3", "params": {"p": 1.0}},
                {"class_name": "albumentations.Normalize", "params": {}},
                {"class_name": "albumentations.pytorch.transforms.ToTensorV2", "params": {"p": 1.0}}
            ]
        }
    },
    "optuna": {
        "use_optuna": True,
        "n_trials": 5,
        "params": {
            "lr": {"min": 1e-5, "max": 1e-3, "type": "loguniform"},
            "batch_size": {"values": [4, 8], "type": "categorical"},
            "gradient_clip_val": {"min": 0.0, "max": 0.3, "type": "float"}
        }
    }
}
cfg: DictConfig = OmegaConf.create(CONFIG_DICT)


def main() -> None:
    """
    Main function to run training, evaluation, and model export.
    """
    set_seed(cfg.training.seed)

    global BASE_SAVE_DIR
    date_folder = datetime.datetime.now().strftime("%Y%m%d")
    BASE_SAVE_DIR = os.path.join(cfg.general.save_dir, date_folder)
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    
    best_model_folder = os.path.join(BASE_SAVE_DIR, "best_model")
    os.makedirs(best_model_folder, exist_ok=True)
    
    metrics_data = []  # For storing evaluation metrics
    
    # ----- Stage 1: Detection (if enabled) -----
    if cfg.stage_detection:
        print("Training Stage 1: Detection (binary classification)...")
        # (Optuna tuning and training logic here)
        # For brevity, assume similar structure as below.
        # Save checkpoint and evaluate.
        pass

    # ----- Stage 2: Classification -----
    if cfg.stage_classification:
        print("Training Stage 2: Classification (multi-class)...")
        # Example: Use single-split training (set cross_validation False in cfg)
        classification_csv = cfg.data.classification_csv
        from bacteria_lib.data import PatchClassificationDataset  # Ensure using library import
        from bacteria_lib.models import build_classifier
        from bacteria_lib.callbacks import PlotMetricsCallback, OptunaReportingCallback
        # (Here you would call the train_stage, continue_training, evaluate_model functions,
        #  which you can import from your library if you package them there.
        #  For this example, we assume they are defined in main.py.)
        
        # For demonstration, we'll call our previously defined train_stage:
        from optuna.exceptions import TrialPruned  # if needed
        
        # --- Train Stage ---
        classification_model, classification_val_acc = train_stage(cfg, classification_csv, num_classes=9, stage_name="classification")
        print(f"Classification Stage Validation Accuracy: {classification_val_acc:.4f}")
        metrics_data.append({"stage": "classification", "val_acc": float(classification_val_acc.item())})
        
        classification_checkpoint = os.path.join(best_model_folder, "best_classification.ckpt")
        torch.save(classification_model.state_dict(), classification_checkpoint)
        print(f"Saved classification best model checkpoint to {classification_checkpoint}")
        
        classification_model = continue_training(classification_model, cfg, classification_csv, num_classes=9, stage_name="classification")
        evaluate_model(classification_model, classification_csv, cfg, stage="Classification")
    
    # Save evaluation metrics for comparison.
    metrics_df = pd.DataFrame(metrics_data)
    eval_table_path = os.path.join(BASE_SAVE_DIR, "evaluation_metrics.csv")
    metrics_df.to_csv(eval_table_path, index=False)
    print(f"Saved evaluation metrics to {eval_table_path}")
    
    if cfg.stage_classification:
        classification_model.load_state_dict(torch.load(classification_checkpoint))
        classification_model.eval()
        scripted_model = torch.jit.script(classification_model)
        export_path = os.path.join(best_model_folder, "best_classification_scripted.pt")
        scripted_model.save(export_path)
        print(f"Exported scripted classification model to {export_path}")
    
    # Optional: Evaluate on dedicated test set.
    if "test_csv" in cfg.data and os.path.exists(cfg.data.test_csv):
        print("Evaluating on the dedicated test set...")
        evaluate_on_test(classification_model, cfg.data.test_csv, cfg)
    
    print("Training finished. Best models and evaluation metrics are saved in:", best_model_folder)


if __name__ == "__main__":
    main()
