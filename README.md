# Neural Decoding ML

## Project Overview
This project aims to **decode the spatial position** of a mouse in a virtual reality corridor, using **neuronal activity** recorded from the **piriform cortex** during an associative task involving odors and contexts.

The aim is to implement and compare different **Machine Learning models** to predict the animal's position based on patterns of neuronal firing.

---

## Project Structure
```plaintext
datasets/              # Processed data (.mat files)
models/                # Model architectures (MLP, RNN, LSTM)
training/              # Training scripts
evaluation/            # Model evaluation scripts
utils/                 # Preprocessing and helper functions
experiments/           # Experiment configurations and runs
results/               # Results and generated plots
README.md              # This file
requirements.txt       # (To be completed later) Project dependencies
```

---

## Workflow Plan

### Stage 1: Baseline Models
- Implement MLP, RNN, and LSTM models.
- Train each model with preprocessed data using fixed hyperparameters.
- Establish baseline performance for each model.
- Evaluate and compare initial results.

### Stage 2: Simple Hyperparameter Optimization
- Perform a limited **Grid Search** over a selected set of hyperparameters.
- Tune parameters such as hidden layer size, number of layers, and learning rate.
- Retrain models and assess improvements over the baseline.

### Stage 3: Cross-Validation and Advanced Optimization
- Implement **k-fold Cross Validation** for robust performance estimates.
- Apply **Bayesian Optimization** for intelligent hyperparameter search.
- Evaluate improvements from fine-tuned configurations.
- Analyze model robustness and generalization.

---

## Evaluation Metrics
During the project, these metrics will be monitored:
- Prediction accuracy.
- Root Mean Square Error (RMSE).
- Predicted vs. actual plots.
- Loss and accuracy curves per epoch.

(More metrics and visualizations may be added later.)

---

## Future Additions
- Experiment configuration files in `.yaml` or `.json`.
- Logging system for experiments and results.
- Automated organization of experiment outputs.
- Saving trained model checkpoints.

---

## Current Status
- Project initialized.
- Basic data preprocessing in progress.
- Workflow plan established and approved.

---

> _"This project is both a technical challenge and an opportunity to deepen the understanding of machine learning models applied to real biological data."_