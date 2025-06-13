# run_palmprint_pipeline.py

"""
This is the main pipeline script:
- Loads dataset
- Trains CNN, ScatClassifier, FusionNet
- Saves training curves, confusion matrix, and top-k accuracy bar plots
"""

# ---- Imports ----
from palmprint_pipeline import PalmprintFusionDataset, PalmprintCNN, ScatClassifier, FusionNet, get_scat_dim
from palmprint_training import train_model, compute_topk_accuracy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from kymatio.torch import Scattering2D
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_training_plots(train_losses, val_losses, train_accs, val_accs, model_name="model"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axs[0].plot(train_losses, label="Train Loss", color="tomato")
    axs[0].plot(val_losses, label="Val Loss", color="royalblue")
    axs[0].set_title("Loss Over Epochs")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # Accuracy
    axs[1].plot(train_accs, label="Train Accuracy", color="seagreen")
    axs[1].plot(val_accs, label="Val Accuracy", color="orange")
    axs[1].set_title("Accuracy Over Epochs")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{VIS_DIR}/{model_name}_loss_acc_plot.png")
    plt.close()

def save_topk_barplot(top1, top3, top5, model_name="model"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    accuracies = [top1, top3, top5]
    labels = ["Top-1", "Top-3", "Top-5"]
    colors = ['cornflowerblue', 'mediumseagreen', 'lightcoral']
    plt.bar(labels, accuracies, color=colors)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title(f"{model_name.upper()} Top-K Accuracy")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"./visualizations/{model_name}_topk_accuracy.png")
    plt.close()


# ---- Configuration ----
DATA_PATH = "/home/mpatteparapu/CS286/BMPD/"
EPOCHS = 10
BATCH_SIZE = 32
IMAGE_SHAPE = (64, 64)
VIS_DIR = "./visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

# ---- Load Dataset ----
print(" Loading dataset...")
dataset = PalmprintFusionDataset(root_dir=DATA_PATH, shape=IMAGE_SHAPE)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
num_classes = len(dataset.label_map)

# ---- Loop over Models ----
for MODEL_TYPE in ["cnn", "scat", "fusion"]:
    print(f"\n\n Running {MODEL_TYPE.upper()} Model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Model
    if MODEL_TYPE == "cnn":
        model = PalmprintCNN(num_classes)
        scattering = None
    elif MODEL_TYPE == "scat":
        scat_dim = get_scat_dim(shape=IMAGE_SHAPE)
        scattering = Scattering2D(J=3, shape=IMAGE_SHAPE).to(device)
        model = ScatClassifier(input_dim=scat_dim, num_classes=num_classes)
    elif MODEL_TYPE == "fusion":
        scat_dim = get_scat_dim(shape=IMAGE_SHAPE)
        scattering = Scattering2D(J=3, shape=IMAGE_SHAPE).to(device)
        model = FusionNet(scat_dim=scat_dim, num_classes=num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # ---- Train and Validate ----
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader,
                                           criterion, optimizer, EPOCHS,
                                           model_type=MODEL_TYPE, scattering=scattering)

    # ---- Save plots ----
    save_training_plots(train_losses, val_losses, train_accuracies, val_accuracies, model_name=MODEL_TYPE)

    print(f"\nTop-K Accuracy for {MODEL_TYPE.upper()}:")
    topk_scores = []
    for k in [1, 3, 5]:
        topk = compute_topk_accuracy(model, val_loader, k=k, model_type=MODEL_TYPE, scattering=scattering)
        print(f"Top-{k} Accuracy: {topk:.4f}")
        topk_scores.append(topk)

    save_topk_barplot(topk_scores[0], topk_scores[1], topk_scores[2], model_name=MODEL_TYPE)

    # ---- Save confusion matrix ----
    true_labels, pred_labels = [], []
    model.eval()
    with torch.no_grad():
        for rgb, gray, yb in val_loader:
            rgb, gray = rgb.to(device), gray.to(device)
            if MODEL_TYPE == "cnn":
                preds = model(rgb)
            elif MODEL_TYPE == "scat":
                feat = scattering(gray).view(gray.size(0), -1)
                preds = model(feat)
            elif MODEL_TYPE == "fusion":
                preds = model(rgb, gray, scattering)
            pred_labels.extend(preds.argmax(1).cpu().numpy())
            true_labels.extend(yb.numpy())

    # save_confusion_matrix(true_labels, pred_labels, model_name=MODEL_TYPE)
