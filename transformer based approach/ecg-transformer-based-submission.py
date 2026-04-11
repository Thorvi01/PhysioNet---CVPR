# ===============================
# Submission Notebook 2
# ===============================

import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn

# ===== 1. Paths =====
MODEL_PATH = '/kaggle/input/datasets/gclling/transformer-ecg-model/transformer_ecg.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== 2. Model  =====
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

# ===== 3. Load model =====
model = SimpleViT().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded")

# ===== 4. Load test metadata =====
BASE_PATH = '/kaggle/input/competitions/physionet-ecg-image-digitization'

test_df = pd.read_csv(f'{BASE_PATH}/test.csv')
TEST_DIR = f'{BASE_PATH}/test'

# sanity check
print(test_df.head())
print("Num test samples:", len(test_df))

# ===== 5. Dummy reconstruction=====
# baseline

submission = []

for _, row in test_df.iterrows():
    base_id = row['id']
    lead = row['lead']
    num_rows = row['number_of_rows']

    # ===== inference =====
    img_path = f"{TEST_DIR}/{base_id}.png"

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))

    img = torch.tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img)
        pred = torch.sigmoid(pred).cpu().numpy()[0,0]

    # ===== dummy signal（ensure submission format）=====
    signal = np.zeros(num_rows)

    for i in range(num_rows):
        submission.append({
            "id": f"{base_id}_{i}_{lead}",
            "value": signal[i]
        })

submission_df = pd.DataFrame(submission)
submission_df.to_csv('/kaggle/working/submission.csv', index=False)

print("submission.csv generated")