#!/usr/bin/env python3
# Trabalho 2 – Tarefa 1
# Classificação MNIST com CNN
# Métricas com sklearn.metrics (Precision, Recall, F1 por classe + macro)
# Matriz de confusão mostrada no fim

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support
)

# ============================================================
# Modelo CNN (base Parte 11)
# ============================================================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 28 -> 14

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)       # 14 -> 7
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ============================================================
# Visualização da matriz de confusão
# ============================================================
def plot_confusion_matrix(cm, show=True, save_path=None):
    classes = list(range(10))

    plt.figure(figsize=(7, 7))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)

    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            plt.text(
                j, i, cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="runs/t1")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("=== Trabalho 2 | Tarefa 1 ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # --------------------------------------------------------
    # Dataset MNIST
    # --------------------------------------------------------
    print("[INFO] Loading MNIST dataset (first run may take a while)...")

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    print("[INFO] MNIST loaded")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --------------------------------------------------------
    # Treino
    # --------------------------------------------------------
    print("[INFO] Training started")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"[INFO] Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss:.4f}")

    # --------------------------------------------------------
    # Guardar modelo
    # --------------------------------------------------------
    model_path = os.path.join(args.out, "best_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved: {model_path}")

    # --------------------------------------------------------
    # Avaliação
    # --------------------------------------------------------
    print("[INFO] Evaluating model...")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())

    # --------------------------------------------------------
    # Métricas (sklearn)
    # --------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)

    precision_c, recall_c, f1_c, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average=None,
        zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    # Mostrar métricas no terminal
    print("\nPer-class metrics:")
    for i in range(10):
        print(
            f"Class {i}: "
            f"P={precision_c[i]:.3f} | "
            f"R={recall_c[i]:.3f} | "
            f"F1={f1_c[i]:.3f}"
        )

    print("\nMacro-average metrics:")
    print(
        f"Precision={precision_macro:.3f} | "
        f"Recall={recall_macro:.3f} | "
        f"F1={f1_macro:.3f}"
    )

    # Guardar métricas em ficheiro
    metrics = {
        "per_class": {
            str(i): {
                "precision": float(precision_c[i]),
                "recall": float(recall_c[i]),
                "f1": float(f1_c[i])
            }
            for i in range(10)
        },
        "macro": {
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1": float(f1_macro)
        }
    }

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --------------------------------------------------------
    # Matriz de confusão (MOSTRAR NO FIM)
    # --------------------------------------------------------
    plot_confusion_matrix(
        cm,
        show=True,
        save_path=os.path.join(args.out, "confusion_matrix.png")
    )

    print("[DONE] Metrics computed and confusion matrix displayed")
    print("[DONE] All outputs in:", args.out)

# ============================================================
if __name__ == "__main__":
    main()
