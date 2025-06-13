# palmprint_training.py

"""
This module defines training, evaluation, confusion matrix and Top-K accuracy calculation
for CNN, Scattering Classifier, and FusionNet models.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from kymatio.torch import Scattering2D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from palmprint_pipeline import PalmprintFusionDataset, PalmprintCNN, ScatClassifier, FusionNet, get_scat_dim

# ============================================
# TRAINING FUNCTION
"""
Unified function to train CNN, ScatClassifier, or FusionNet dynamically
Key Actions:
Loop over training epochs (default: 10)
Forward pass:
If CNN: feed RGB to CNN.

If Scattering: feed Grayscale to Scattering + ScatClassifier

If Fusion: feed both RGB and Grayscale to FusionNet

Calculate CrossEntropyLoss

Backward pass → Update model parameters

Track training loss and validation accuracy every epoch
"""
# ============================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, model_type="cnn", scattering=None):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for rgb, gray, yb in train_loader:
            rgb, gray, yb = rgb.to(device), gray.to(device), yb.to(device)
            optimizer.zero_grad()

            if model_type == "cnn":
                preds = model(rgb)
            elif model_type == "scat":
                with torch.no_grad():
                    feat = scattering(gray).view(gray.size(0), -1)
                preds = model(feat)
            elif model_type == "fusion":
                preds = model(rgb, gray, scattering)

            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (preds.argmax(1) == yb).sum().item()
            total += yb.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct / total)

        # Validation phase
        val_loss = 0
        correct_val = 0
        total_val = 0
        model.eval()
        with torch.no_grad():
            for rgb, gray, yb in val_loader:
                rgb, gray, yb = rgb.to(device), gray.to(device), yb.to(device)
                if model_type == "cnn":
                    outputs = model(rgb)
                elif model_type == "scat":
                    feat = scattering(gray).view(gray.size(0), -1)
                    outputs = model(feat)
                elif model_type == "fusion":
                    outputs = model(rgb, gray, scattering)

                val_loss += criterion(outputs, yb).item()
                correct_val += (outputs.argmax(1) == yb).sum().item()
                total_val += yb.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct_val / total_val)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies


# ============================================
# EVALUATION FUNCTION
"""
Purpose:
Calculate validation accuracy at the end of each epoch

Key Actions:

Model switched to evaluation mode (model.eval())

Predictions collected and compared to true labels

Accuracy calculated (percentage of correct predictions)
"""
# ============================================

def evaluate_model(model, val_loader, model_type, scattering):
    """Evaluate the model on validation data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for rgb, gray, yb in val_loader:
            rgb, gray = rgb.to(device), gray.to(device)
            if model_type == "cnn":
                outputs = model(rgb)
            elif model_type == "scat":
                feat = scattering(gray).view(gray.size(0), -1)
                outputs = model(feat)
            elif model_type == "fusion":
                outputs = model(rgb, gray, scattering)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(yb.numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return acc

# ============================================
# CONFUSION MATRIX PLOTTING
# ============================================

def plot_confusion_matrix(true_labels, pred_labels, class_names=None, title="Confusion Matrix"):
    """Display confusion matrix."""
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title(title)
    plt.show()

# ============================================
# TOP-K ACCURACY
# ============================================

def compute_topk_accuracy(model, dataloader, k=3, model_type="cnn", scattering=None):
    """Compute Top-k prediction accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct_topk = 0
    total = 0

    with torch.no_grad():
        for rgb, gray, yb in dataloader:
            rgb, gray, yb = rgb.to(device), gray.to(device), yb.to(device)

            if model_type == "cnn":
                outputs = model(rgb)
            elif model_type == "scat":
                feat = scattering(gray).view(gray.size(0), -1)
                outputs = model(feat)
            elif model_type == "fusion":
                outputs = model(rgb, gray, scattering)

            topk_preds = outputs.topk(k=k, dim=1).indices
            match = topk_preds.eq(yb.view(-1, 1)).sum().item()
            correct_topk += match
            total += yb.size(0)

    return correct_topk / total

