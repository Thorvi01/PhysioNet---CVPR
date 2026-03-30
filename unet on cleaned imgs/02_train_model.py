# Cell 1: Verify data + overlay check
import os, cv2, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

DATA_DIR = '/kaggle/input/datasets/tylerde/ecg-training-data-0001-only/data'
KAGGLE_DIR = '/kaggle/input/competitions/physionet-ecg-image-digitization'
device = 'cuda:0'

def load_sparse_mask(filepath):
    data = np.load(filepath)
    shape = tuple(data['shape'])
    mask = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]):
        mask[i, data[f'ch{i}_y'], data[f'ch{i}_x']] = data[f'ch{i}_v']
    return mask

fold_df = pd.read_csv(f'{DATA_DIR}/train_fold.csv', dtype={'id': str, 'type_id': str})
print(f'Total samples: {len(fold_df)}')

ZERO_MV = [703.5, 987.5, 1271.5, 1531.5]
W_SIZE = 240

# Overlay mask on rectified image
sid = str(fold_df.iloc[0]['id'])
img = cv2.imread(f'{DATA_DIR}/rectified/{sid}-0001.rect.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (5600, 1700), interpolation=cv2.INTER_LINEAR)

full_mask = load_sparse_mask(f'{DATA_DIR}/masks/{sid}.mask-coo.npz')

fig, axes = plt.subplots(4, 1, figsize=(20, 16))
for row_idx in range(4):
    zmv = int(ZERO_MV[row_idx])
    h0, h1 = zmv - W_SIZE, zmv + W_SIZE
    
    row_img = img[h0:h1, :, :].copy()
    m = full_mask[row_idx, h0:h1, :]
    H, W = m.shape
    y_idx = np.arange(H, dtype=np.float32)[:, None]
    denom = m.sum(axis=0) + 1e-8
    y_pos = (m * y_idx).sum(axis=0) / denom
    
    for x in range(301, 5301):
        y = int(round(y_pos[x]))
        if 0 <= y < H:
            row_img[max(0,y-3):min(H,y+4), x] = [0, 0, 255]
    
    axes[row_idx].imshow(row_img)
    axes[row_idx].set_title(f'Row {row_idx}: rectified + blue mask overlay')

plt.suptitle(f'Sample {sid}: Does blue line follow the trace?', fontsize=14)
plt.tight_layout()
plt.show()

# Cell 2: Split: train - test 
def clean_image(img):
    r, g = img[:,:,0].astype(float), img[:,:,1].astype(float)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (gray < 120) & ((r - g) < 10)
    cleaned = np.full_like(img, 255)
    cleaned[mask] = img[mask]
    return cleaned

ZERO_MV = [703.5, 987.5, 1271.5, 1531.5]
WINDOW_SIZE = 240
IMG_W = 5600

class ECGRowDataset(Dataset):
    def __init__(self, df, data_dir, is_train=True):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.is_train = is_train
        self.samples = []
        for idx in range(len(df)):
            for row_idx in range(4):
                self.samples.append((idx, row_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_idx, row_idx = self.samples[idx]
        row = self.df.iloc[img_idx]
        sid = str(row['id'])
        var = str(row['type_id'])
        
        img = cv2.imread(f'{self.data_dir}/rectified/{sid}-{var}.rect.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_W, 1700), interpolation=cv2.INTER_LINEAR)
        img = clean_image(img)
        
        zmv = ZERO_MV[row_idx]
        h0, h1 = int(zmv) - WINDOW_SIZE, int(zmv) + WINDOW_SIZE
        H = img.shape[0]
        src_h0, src_h1 = max(0, h0), min(H, h1)
        dst_h0 = src_h0 - h0
        dst_h1 = dst_h0 + (src_h1 - src_h0)
        
        row_img = np.full((WINDOW_SIZE*2, IMG_W, 3), 255, dtype=np.uint8)
        row_img[dst_h0:dst_h1] = img[src_h0:src_h1]
        
        full_mask = load_sparse_mask(f'{self.data_dir}/masks/{sid}.mask-coo.npz')
        row_mask = np.zeros((WINDOW_SIZE*2, IMG_W), dtype=np.float32)
        row_mask[dst_h0:dst_h1] = full_mask[row_idx][src_h0:src_h1]
        
        if self.is_train and np.random.random() > 0.5:
            row_img = np.fliplr(row_img).copy()
            row_mask = np.fliplr(row_mask).copy()
        
        row_img = torch.from_numpy(row_img.transpose(2, 0, 1)).float() / 255.0
        row_mask = torch.from_numpy(row_mask).unsqueeze(0).float()
        return row_img, row_mask

val_fold = 0
df_train = fold_df[fold_df['fold'] != val_fold].reset_index(drop=True)
df_val = fold_df[fold_df['fold'] == val_fold].reset_index(drop=True)

train_dataset = ECGRowDataset(df_train, DATA_DIR, is_train=True)
val_dataset = ECGRowDataset(df_val, DATA_DIR, is_train=False)
print(f'Train: {len(train_dataset)} rows, Val: {len(val_dataset)} rows')

# Cell 3: Simple UNet
!pip install segmentation-models-pytorch -q

import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name='resnet18',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
).to(device)

print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

# Cell 4: Training
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
scaler = torch.amp.GradScaler('cuda')
pos_weight = torch.tensor([20.0]).to(device)

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0
    for i, (img, mask) in enumerate(loader):
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            pred = model(img)
            loss = F.binary_cross_entropy_with_logits(pred, mask, pos_weight=pos_weight)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if (i+1) % 200 == 0:
            print(f'    batch {i+1}/{len(loader)}, loss: {total_loss/(i+1):.4f}')
    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            with torch.amp.autocast('cuda'):
                pred = model(img)
                loss = F.binary_cross_entropy_with_logits(pred, mask, pos_weight=pos_weight)
            total_loss += loss.item()
    return total_loss / len(loader)

NUM_EPOCHS = 20
best_val_loss = float('inf')
history = []

for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch+1}/{NUM_EPOCHS} — train: {train_loss:.4f}, val: {val_loss:.4f}, lr: {lr:.6f}')
    history.append({'epoch': epoch+1, 'train': train_loss, 'val': val_loss})
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '/kaggle/working/best_row_model.pth')
        print(f'  ★ Saved')

print(f'\nBest val_loss: {best_val_loss:.4f}')
plt.plot([h['epoch'] for h in history], [h['train'] for h in history], label='train')
plt.plot([h['epoch'] for h in history], [h['val'] for h in history], label='val')
plt.legend(); plt.title('Loss'); plt.show()

# Cell 5: Compute SNR
from scipy.signal import fftconvolve

model.load_state_dict(torch.load('/kaggle/working/best_row_model.pth', map_location=device))
model.eval()

T0, T1 = 301, 5301
MV_TO_PIXEL = 79.0
ZERO_MV_TRIM = [287.5, 571.5, 855.5, 1115.5]  # trimmed coords
LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
ROW_TO_LEADS = [['I','aVR','V1','V4'],['II','aVL','V2','V5'],['III','aVF','V3','V6']]

train_meta = pd.read_csv(f'{KAGGLE_DIR}/train.csv')
train_meta['id'] = train_meta['id'].astype(str)

def compute_snr_fast(pred_by_lead, gt_by_lead, fs):
    total_sig, total_noise = 0.0, 0.0
    max_shift = int(0.2 * fs)
    for lead in pred_by_lead:
        if lead not in gt_by_lead: continue
        pred = pred_by_lead[lead].astype(np.float64)
        gt = gt_by_lead[lead].astype(np.float64)
        corr = fftconvolve(gt, pred[::-1], mode='full')
        mid = len(gt) - 1
        search = corr[mid-max_shift:mid+max_shift+1]
        shift = np.argmax(search) - max_shift
        if shift > 0: p, g = pred[shift:], gt[:len(pred)-shift]
        elif shift < 0: g, p = gt[-shift:], pred[:len(gt)+shift]
        else: p, g = pred, gt
        ml = min(len(p), len(g)); p, g = p[:ml], g[:ml]
        p = p + (np.mean(g) - np.mean(p))
        total_sig += np.sum(g**2)
        total_noise += np.sum((g - p)**2)
    if total_noise < 1e-12: return 60.0
    return 10 * np.log10(total_sig / total_noise)

# Evaluate on val set
snrs = []
for img_idx in range(min(20, len(df_val))):
    sid = str(df_val.iloc[img_idx]['id'])
    row_meta = train_meta[train_meta['id'] == sid].iloc[0]
    fs = row_meta['fs']
    length_ii = int(np.floor(fs * 10))
    
    # Predict all 4 rows
    all_series = []
    for row_idx in range(4):
        sample_idx = img_idx * 4 + row_idx
        img, _ = val_dataset[sample_idx]
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                pred = torch.sigmoid(model(img.unsqueeze(0).to(device))).cpu().numpy()[0, 0]
        
        # Weighted average for y-position
        H, W = pred.shape
        y_idx = np.arange(H, dtype=np.float32)[:, None]
        denom = pred.sum(axis=0) + 1e-8
        y_pos = (pred * y_idx).sum(axis=0) / denom
        
        # Crop to signal region and convert to mV
        y_signal = y_pos[T0:T1]
        mv = (WINDOW_SIZE - y_signal) / MV_TO_PIXEL  # center of crop = 0 mV
        
        if len(mv) != length_ii:
            mv = scipy_signal.resample(mv, length_ii).astype(np.float32)
        all_series.append(mv)
    
    # Split rows into 12 leads
    gt_df = pd.read_csv(f'{KAGGLE_DIR}/train/{sid}/{sid}.csv')
    gt_by_lead = {ld: gt_df[ld].dropna().values.astype(np.float32) for ld in LEAD_NAMES}
    
    pred_by_lead = {}
    for l in range(3):
        leads = ROW_TO_LEADS[l]
        lengths = [len(gt_by_lead[ld]) for ld in leads]
        if leads[0] == 'II': lengths[0] = len(all_series[l]) - sum(lengths[1:])
        idx_split = np.cumsum(lengths)[:-1]
        for k, s in zip(leads, np.split(all_series[l], idx_split)):
            pred_by_lead[k] = s
    pred_by_lead['II'] = all_series[3]
    
    for ld in LEAD_NAMES:
        gl = len(gt_by_lead[ld])
        p = pred_by_lead.get(ld, np.zeros(gl))
        pred_by_lead[ld] = np.concatenate([p, np.zeros(gl)])[:gl]
    
    snr = compute_snr_fast(pred_by_lead, gt_by_lead, fs)
    snrs.append(snr)
    print(f'  {sid} (fs={fs}): {snr:.2f} dB')

print(f'\nMean SNR: {np.mean(snrs):.2f} dB')
print(f'Median SNR: {np.median(snrs):.2f} dB')
print(f'\nReference: their single model = 23.10 dB')