# Notebook 1
# this is the transformer model training file running on kaggle
# the input includes the pre-processed dataset

import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===== 1. Check dataset =====
print("===== DATA STRUCTURE =====")
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)

# ===== 2. Correct path =====
BASE_DIR = '/kaggle/input/datasets/tylerde/ecg-training-data-0001-only/data'

IMG_DIR = os.path.join(BASE_DIR, 'rectified')
MASK_DIR = os.path.join(BASE_DIR, 'masks')

# ===== 3. Build dataframe =====
image_files = sorted(os.listdir(IMG_DIR))
mask_files = sorted(os.listdir(MASK_DIR))

def get_id_from_img(x):
    return x.split('-')[0]

def get_id_from_mask(x):
    return x.split('.')[0]

img_map = {get_id_from_img(f): f for f in image_files}
mask_map = {get_id_from_mask(f): f for f in mask_files}

common_ids = sorted(list(set(img_map.keys()) & set(mask_map.keys())))

df = pd.DataFrame({
    'id': common_ids
})

print("Samples:", len(df))

# ===== 4. Load COO mask =====
def load_coo_mask(npz_path):
    data = np.load(npz_path)

    C, H, W = data['shape']
    mask = np.zeros((C, H, W), dtype=np.float32)

    for i in range(C):
        ys = data[f'ch{i}_y']
        xs = data[f'ch{i}_x']
        vs = data[f'ch{i}_v']
        mask[i, ys, xs] = vs

    return mask

# ===== 5. Dataset =====
class ECGDataset(Dataset):
    def __init__(self, df, img_dir, mask_dir, img_size=512):
        self.df = df
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sid = self.df.iloc[idx]['id']

        img_path = os.path.join(self.img_dir, f"{sid}-0001.rect.png")
        mask_path = os.path.join(self.mask_dir, f"{sid}.mask-coo.npz")

        # ===== image =====
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # ===== mask (COO → dense) =====
        mask = load_coo_mask(mask_path)
        mask = mask[0]

        # ===== resize =====
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        # ===== format =====
        img = np.transpose(img, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# ===== 6. DataLoader =====
dataset = ECGDataset(df, IMG_DIR, MASK_DIR)

train_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2
)

# ===== 7. Model =====
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=512, patch_size=16, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class SimpleViT(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch = PatchEmbedding()

        num_patches = (512 // 16) ** 2
        self.pos = nn.Parameter(torch.randn(1, num_patches, 256))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, 4)

        self.decoder = nn.ConvTranspose2d(256, 1, 16, 16)

    def forward(self, x):
        x = self.patch(x)
        x = x + self.pos

        x = self.transformer(x)

        B, N, C = x.shape
        H = W = int(N ** 0.5)

        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x = self.decoder(x)

        return x

# ===== 8. Train =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleViT().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(3):
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# ===== 9. Save =====
torch.save(model.state_dict(), '/kaggle/working/transformer_ecg.pth')

print("DONE")