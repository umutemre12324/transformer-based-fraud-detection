# Fraud Detection with Transformer

This project demonstrates a synthetic fraud detection pipeline using a Transformer-based classifier on sequential behavioral data. It includes a data generation script, preprocessing logic, model definition, and training workflow.

## Overview

The dataset is generated from simulated user behavior sequences with five features over 20 time steps:

- click_count
- page_duration
- amount_spent
- scroll_depth
- cart_events

Three classes are used:

- 0: normal users
- 1: suspicious behavior
- 2: fraudulent behavior

The model uses a Transformer encoder followed by a classification head to learn temporal patterns in user activity.

## Project Structure

```text
.
├── README.md
├── dataset.py
├── model.py
├── preprocessing.py
├── train.py
├── X.npy
├── y.npy
├── dataset.csv
├── scaler.pkl
├── artifacts/
│   ├── best_model.pth
│   ├── scaler.pkl
│   └── metrics.json
└── venv/
```

## Requirements

This project uses Python and the following packages:

- Python 3.9+
- NumPy
- pandas
- scikit-learn
- PyTorch
- joblib

Install dependencies with:

```bash
pip install numpy pandas scikit-learn torch joblib
```

If you are using the included virtual environment, activate it first:

```bash
venv\Scripts\activate
```

## Data Generation

The synthetic dataset is created via `preprocessing.py` (or `dataset.py`, depending on the workflow used).

To generate the dataset:

```bash
python preprocessing.py
```

This creates:

- `X.npy`
- `y.npy`
- `dataset.csv`

## Training

Train the model with:

```bash
python train.py
```

Optional training arguments:

```bash
python train.py --epochs 100 --batch-size 32 --learning-rate 0.001 --seed 42
```

The training script:

- loads `X.npy` and `y.npy`
- splits the data into train/validation/test sets
- scales features using the training split only
- trains the Transformer model
- saves the best model checkpoint
- prints test metrics
- writes metrics to `artifacts/metrics.json`

## Model

The Transformer architecture is implemented in `model.py` and includes:

- input projection layer
- Transformer encoder stack
- mean pooling across time steps
- classification MLP

## Outputs

Training artifacts are stored in the `artifacts/` directory:

- `best_model.pth`: best model checkpoint based on validation loss
- `scaler.pkl`: fitted scaler used for feature normalization
- `metrics.json`: evaluation metrics such as test loss and accuracy

## Notes

- The project uses a deterministic seed (`42`) for reproducibility.
- The model is trained on synthetic data, so it is intended for experimentation and educational purposes rather than production deployment.
- If the dataset files are missing, run `preprocessing.py` before training.

## Example Run

```bash
python preprocessing.py
python train.py --epochs 20
```

This will generate the dataset and train a quick baseline model.
