# palmprint_pipeline.py

"""
This module builds the dataset and model architecture components:
- PalmprintFusionDataset: loads and preprocesses palmprint images
- PalmprintCNN: a basic CNN model for palmprint recognition
- ScatClassifier: classification based on Scattering2D features
- FusionNet: combining CNN features + Scattering features for improved performance
"""

# ---- Import Libraries ----
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from kymatio.torch import Scattering2D
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image

# ============================================
# DATA LOADING + CLEANING + AUGMENTATION
# ============================================

class PalmprintFusionDataset(Dataset):
    """Dataset class to load palmprint images, generate RGB + Grayscale tensors."""

    def __init__(self, root_dir, shape=(64, 64), transform=None):
        """
        Args:
            root_dir (str): Path to dataset directory organized by folders (one per subject).
            shape (tuple): Output size (height, width) to resize images.
            transform (callable, optional): Optional transform for RGB image.
        """
        self.root_dir = root_dir
        self.shape = shape
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {}
        label_counter = 0

        # Walk through each subject folder
        for subject in sorted(os.listdir(root_dir)):
            subject_path = os.path.join(root_dir, subject)
            if not os.path.isdir(subject_path):
                continue

            # Assign a label index to each subject
            if subject not in self.label_map:
                self.label_map[subject] = label_counter
                label_counter += 1

            # Collect all image paths and corresponding labels
            for file in os.listdir(subject_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(subject_path, file))
                    self.labels.append(self.label_map[subject])

    def __len__(self):
        """Return total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Retrieve the (RGB image tensor, grayscale tensor, label) for a sample."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ROI (Region of Interest) extraction
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y_, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            gray = gray[y_:y_+h, x:x+w]
            img = img[y_:y_+h, x:x+w]

        # Resize both RGB and grayscale images
        gray = cv2.resize(gray, self.shape)
        rgb = cv2.resize(img, self.shape)

        # Normalize grayscale to [0,1] and expand dimension
        gray_tensor = torch.tensor(gray / 255.0, dtype=torch.float32).unsqueeze(0)

        # Normalize RGB to [0,1] and permute to channel-first
        rgb_tensor = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)

        # Apply transform if needed
        if self.transform:
            rgb_tensor = self.transform(Image.fromarray(rgb))

        return rgb_tensor, gray_tensor, label

# ============================================
# MODEL DEFINITIONS
# ============================================

class PalmprintCNN(nn.Module):
    """A simple CNN baseline model for classification."""

    def __init__(self, num_classes):
        super(PalmprintCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """Forward pass through CNN."""
        return self.classifier(self.features(x))

class ScatClassifier(nn.Module):
    """Classifier for Scattering2D features."""

    def __init__(self, input_dim, num_classes):
        super(ScatClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Forward pass through MLP after scattering."""
        return self.fc(x)

class FusionNet(nn.Module):
    """Combines CNN features + ScatterNet features (Fusion architecture)."""

    def __init__(self, scat_dim, num_classes):
        super(FusionNet, self).__init__()

        # CNN feature extractor
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Output is 4x4
        )
        self.cnn_flatten = nn.Flatten()
        self.cnn_feat_dim = 128 * 4 * 4

        # Batch Normalization on extracted features
        self.bn_cnn = nn.BatchNorm1d(self.cnn_feat_dim)
        self.bn_scat = nn.BatchNorm1d(scat_dim)

        # Fusion and classification head
        self.fusion = nn.Sequential(
            nn.Linear(self.cnn_feat_dim + scat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb_input, gray_input, scattering):
        """Fusion forward pass."""
        cnn_feat = self.cnn_branch(rgb_input)
        cnn_feat = self.cnn_flatten(cnn_feat)
        cnn_feat = self.bn_cnn(cnn_feat)

        with torch.no_grad():
            scat_feat = scattering(gray_input).view(gray_input.size(0), -1)
        scat_feat = self.bn_scat(scat_feat)

        fused_feat = torch.cat([cnn_feat, scat_feat], dim=1)
        return self.fusion(fused_feat)

# ============================================
# UTILITY: Get scattering output dimension
# ============================================

def get_scat_dim(shape=(64, 64), J=3):
    """Utility function to compute the dimension of scattering features."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scattering = Scattering2D(J=J, shape=shape).to(device)
    dummy = torch.zeros(1, 1, *shape).to(device)
    return scattering(dummy).view(1, -1).shape[1]
