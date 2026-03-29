"""
Input:
    Preprocessed ECG images and corresponding segmentation masks
    generated from Stage 1.

Data format:
    Image:
        shape = (1, 512, 1024)
        range = [0, 1]

    Mask:
        shape = (1, 512, 1024)
        values = {0, 1}

Output:
    Trained Transformer segmentation model (.pth)
"""

# =========================
# 1. Import Libraries
# =========================
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader


# =========================
# 2. Dataset Definition
# =========================
class ECGSegmentationDataset(Dataset):
    """
    Custom Dataset for ECG segmentation.

    Each sample consists of:
        - image: ECG grayscale image
        - mask : binary segmentation mask

    Both are stored as .npy files.
    """

    def __init__(self, image_dir, mask_dir):
        """
        Args:
            image_dir (str): Directory containing ECG images
            mask_dir  (str): Directory containing segmentation masks
        """

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Load and sort filenames to ensure correct pairing
        image_files = sorted(os.listdir(image_dir))
        mask_files = sorted(os.listdir(mask_dir))

        # Ensure dataset consistency
        assert len(image_files) == len(mask_files), \
            "Mismatch between number of images and masks!"

        self.files = image_files

    def __len__(self):
        """
        Returns:
            Total number of samples in dataset
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load a single sample.

        Steps:
            1. Load image (.npy)
            2. Load mask (.npy)
            3. Convert to PyTorch tensors
        """
        filename = self.files[idx]

        image_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        # Load numpy arrays
        image = np.load(image_path)
        mask = np.load(mask_path)

        # Convert to float tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask


# =========================
# 3. Patch Embedding Layer
# =========================
class PatchEmbedding(nn.Module):
    """
    Converts input image into patch embeddings.

    Mechanism:
        - Split image into non-overlapping patches
        - Use Conv2D to project patches into embedding space

    Output:
        Shape: (batch, num_patches, embed_dim)
    """

    def __init__(self, img_size=512, patch_size=16,
                 in_channels=1, embed_dim=256):

        super().__init__()

        # Conv2D acts as patch extractor + linear projection
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: Input image tensor (B, C, H, W)

        Returns:
            Patch embeddings (B, N, D)
        """

        # Apply patch projection
        x = self.projection(x)

        # Flatten spatial dimensions
        x = x.flatten(2)

        # Convert to (batch, num_patches, embed_dim)
        x = x.transpose(1, 2)

        return x


# =========================
# 4. Transformer Model
# =========================
class TransformerSegmentationModel(nn.Module):
    """
    Vision Transformer-based segmentation model.

    Architecture:
        Image → Patch Embedding → Transformer Encoder
              → Linear Decoder → Segmentation Mask
    """

    def __init__(self):
        super().__init__()

        # Patch embedding layer
        self.patch_embed = PatchEmbedding()

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, 2048, 256)
        )

        # Transformer encoder block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

        # Linear decoder: maps embedding → pixel patch
        self.decoder = nn.Linear(256, 16 * 16)

    def forward(self, x):
        """
        Forward pass.

        Steps:
            1. Convert image → patches
            2. Add positional encoding
            3. Pass through Transformer
            4. Decode patches back to image
        """

        # Patch embedding
        patches = self.patch_embed(x)

        # Add positional encoding
        patches = patches + self.pos_embed

        # Transformer expects (sequence_len, batch, dim)
        patches = patches.permute(1, 0, 2)

        # Apply transformer encoder
        encoded = self.transformer(patches)

        # Convert back to (batch, sequence, dim)
        encoded = encoded.permute(1, 0, 2)

        # Decode each patch
        decoded = self.decoder(encoded)

        # Reshape to image
        batch = x.shape[0]

        decoded = decoded.view(batch, 32, 64, 16, 16)
        decoded = decoded.permute(0, 1, 3, 2, 4)

        # Final output shape: (B, 1, 512, 1024)
        decoded = decoded.reshape(batch, 1, 512, 1024)

        return decoded


# =========================
# 5. Training Setup
# =========================

# Device configuration (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and DataLoader
dataset = ECGSegmentationDataset(
    "../data/images",
    "../data/masks"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

# Initialize model
model = TransformerSegmentationModel().to(device)

# Binary segmentation loss
loss_fn = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)


# =========================
# 6. Training Loop
# =========================
EPOCHS = 20

for epoch in range(EPOCHS):

    total_loss = 0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        preds = model(images)

        # Compute loss
        loss = loss_fn(preds, masks)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch: {epoch} | Loss: {total_loss / len(loader):.6f}")


# =========================
# 7. Save Model
# =========================
torch.save(
    model.state_dict(),
    "transformer_ecg_segmentation.pth"
)


# =========================
# 8. Inference Notes
# =========================
"""
During inference:

    preds = model(images)
    prob_mask = torch.sigmoid(preds)

The output represents probability maps:
    shape = (1, 512, 1024)

Each pixel indicates likelihood of ECG waveform.

NOTE:
Segmentation quality directly impacts reconstruction accuracy.
"""