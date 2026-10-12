import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import Transformer


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fraud detection transformer model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_dataset(data_dir: Path):
    x_path = data_dir / "X.npy"
    y_path = data_dir / "y.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Dataset files were not found. Run preprocessing.py first. "
            f"Expected: {x_path} and {y_path}"
        )

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)

    if X.ndim != 3 or X.shape[1] != 5 or X.shape[2] != 20:
        raise ValueError(
            "Unexpected data shape. Expected (n_samples, 5, 20), "
            f"got {X.shape}."
        )

    return X, y


def scale_data(data, scaler: StandardScaler, fit: bool = False):
    n_samples, n_features, n_steps = data.shape
    reshaped = data.transpose(0, 2, 1).reshape(-1, n_features)

    if fit:
        transformed = scaler.fit_transform(reshaped)
    else:
        transformed = scaler.transform(reshaped)

    return transformed.reshape(n_samples, n_steps, n_features).transpose(0, 2, 1).astype(np.float32)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    print("Device:", device)

    data_dir = Path(args.data_dir).resolve()
    X, y = load_dataset(data_dir)

    if len(np.unique(y)) < 2:
        raise ValueError("At least two classes are required for training.")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.seed, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.1875,
        random_state=args.seed,
        stratify=y_trainval,
    )

    scaler = StandardScaler()
    X_train = scale_data(X_train, scaler=scaler, fit=True)
    X_val = scale_data(X_val, scaler=scaler, fit=False)
    X_test = scale_data(X_test, scaler=scaler, fit=False)

    joblib.dump(scaler, ARTIFACT_DIR / "scaler.pkl")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    model = Transformer(
        input_dim=5,
        model_dim=64,
        num_head=4,
        num_layer=2,
        num_classes=int(y.max()) + 1,
        dropout=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == targets).sum().item()
            train_total += targets.size(0)

        train_loss /= max(train_total, 1)
        train_accuracy = train_correct / max(train_total, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == targets).sum().item()
                val_total += targets.size(0)

        val_loss /= max(val_total, 1)
        val_accuracy = val_correct / max(val_total, 1)
        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ARTIFACT_DIR / "best_model.pth")
            print("Best model saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            break

    checkpoint_path = ARTIFACT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            test_correct += (outputs.argmax(1) == targets).sum().item()
            test_total += targets.size(0)

    test_loss /= max(test_total, 1)
    test_accuracy = test_correct / max(test_total, 1)

    metrics = {
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n========== TEST RESULTS ==========")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
