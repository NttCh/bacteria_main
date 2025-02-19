# bacteria_main

**bacteria_main** is the main application that uses the **bacteria_lib** package to train, evaluate, and export bacteria classification models.

## Overview

This repository orchestrates the full training pipeline for bacteria classification. It:
- Loads configuration from a YAML file.
- Sets up the training pipeline using modules from `bacteria_lib`.
- Trains models (with options for single split or cross-validation, and hyperparameter tuning via Optuna).
- Evaluates the models, generating confusion matrices and classification reports.
- Exports the final model (e.g., as a TorchScript file) for deployment.
- Optionally evaluates on a dedicated test set.

## Setup & Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/bacteria_main.git
   cd bacteria_main
