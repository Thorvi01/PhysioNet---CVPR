# Notebook 1: Prepare training data (rectified images + fixed masks + fold CSV)
!pip install connected-components-3d -q

import os, sys, shutil
import cv2
import numpy as np
import pandas as pd
import torch
from scipy import signal as scipy_signal
from sklearn.model_selection import KFold

sys.path.append('/kaggle/input/datasets/hengck23/hengck23-demo-submit-physionet')
import stage0_common
from stage0_model import Net as Stage0Net
import stage1_common
from stage1_model import Net as Stage1Net

KAGGLE_DIR = '/kaggle/input/competitions/physionet-ecg-image-digitization'
WEIGHT_DIR = '/kaggle/input/datasets/hengck23/hengck23-demo-submit-physionet/weight'
OUT_DIR = '/kaggle/working/data'
RECT_DIR = f'{OUT_DIR}/rectified'
MASK_DIR = f'{OUT_DIR}/masks'
ECG_DIR = f'{OUT_DIR}/ecg_csv'
os.makedirs(RECT_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(ECG_DIR, exist_ok=True)

device = 'cuda:0'
train_df = pd.read_csv(f'{KAGGLE_DIR}/train.csv')
train_df['id'] = train_df['id'].astype(str)
train_ids = train_df['id'].tolist()
print(f'Records: {len(train_ids)}')

# === Step 1: Rectify 0001 images through Stage 0+1 ===
print('\n=== Step 1: Rectifying images ===')
s0 = Stage0Net(pretrained=False)
s0 = stage0_common.load_net(s0, f'{WEIGHT_DIR}/stage0-last.checkpoint.pth')
s0.to(device)
s1 = Stage1Net(pretrained=False)
s1 = stage1_common.load_net(s1, f'{WEIGHT_DIR}/stage1-last.checkpoint.pth')
s1.to(device)

fail_ids = []
for i, sid in enumerate(train_ids):
    out_path = f'{RECT_DIR}/{sid}-0001.rect.png'
    if os.path.exists(out_path):
        continue
    try:
        image = cv2.imread(f'{KAGGLE_DIR}/train/{sid}/{sid}-0001.png', cv2.IMREAD_COLOR_RGB)
        batch = stage0_common.image_to_batch(image)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            with torch.no_grad():
                out = s0(batch)
                rot, kp = stage0_common.output_to_predict(image, batch, out)
                norm, kp, homo = stage0_common.normalise_by_homography(rot, kp)
        batch1 = {'image': torch.from_numpy(np.ascontiguousarray(norm.transpose(2,0,1))).unsqueeze(0)}
        with torch.amp.autocast('cuda', dtype=torch.float16):
            with torch.no_grad():
                out1 = s1(batch1)
                gp, more = stage1_common.output_to_predict(norm, batch1, out1)
                rect = stage1_common.rectify_image(norm, gp)
        cv2.imwrite(out_path, cv2.cvtColor(rect, cv2.COLOR_RGB2BGR))
    except:
        fail_ids.append(sid)
    torch.cuda.empty_cache()
    if (i+1) % 100 == 0:
        print(f'  {i+1}/{len(train_ids)} done, fails: {len(fail_ids)}')

del s0, s1; torch.cuda.empty_cache()
print(f'  Done: {len(os.listdir(RECT_DIR))} images, Failed: {len(fail_ids)}')

# === Step 2: Generate fixed masks ===
print('\n=== Step 2: Generating masks ===')
MASK_H, MASK_W = 1700, 5600
T0, T1, SIGNAL_W = 301, 5301, 5000
ZERO_MV = [703.5, 987.5, 1271.5, 1531.5]
MV_TO_PIXEL = 79.0
ROW_TO_LEADS = [['I','aVR','V1','V4'], ['II','aVL','V2','V5'], ['III','aVF','V3','V6']]

def create_mask(ecg_csv_path):
    ecg_df = pd.read_csv(ecg_csv_path)
    mask = np.zeros((4, MASK_H, MASK_W), dtype=np.float32)
    for row_idx in range(3):
        leads = ROW_TO_LEADS[row_idx]
        zero_y = ZERO_MV[row_idx]
        row_values = []
        for lead_name in leads:
            row_values.append(ecg_df[lead_name].dropna().values.astype(np.float32))
        # Fix: Lead II in Row 1 — use only first 2.5s (same length as Lead I)
        if leads[0] == 'II':
            ii_short_len = len(ecg_df['I'].dropna())
            row_values[0] = row_values[0][:ii_short_len]
        all_values = np.concatenate(row_values)
        if len(all_values) != SIGNAL_W:
            all_values = scipy_signal.resample(all_values, SIGNAL_W).astype(np.float32)
        y_positions = zero_y - all_values * MV_TO_PIXEL
        for col_idx in range(SIGNAL_W):
            y = y_positions[col_idx]
            if y < 0 or y >= MASK_H - 1: continue
            yf = int(np.floor(y)); frac = y - yf; x = T0 + col_idx
            mask[row_idx, yf, x] = 1.0 - frac
            mask[row_idx, yf + 1, x] = frac
    # Row 3: Lead II full 10s
    zero_y = ZERO_MV[3]
    lead_ii = ecg_df['II'].dropna().values.astype(np.float32)
    if len(lead_ii) != SIGNAL_W:
        lead_ii = scipy_signal.resample(lead_ii, SIGNAL_W).astype(np.float32)
    y_positions = zero_y - lead_ii * MV_TO_PIXEL
    for col_idx in range(SIGNAL_W):
        y = y_positions[col_idx]
        if y < 0 or y >= MASK_H - 1: continue
        yf = int(np.floor(y)); frac = y - yf; x = T0 + col_idx
        mask[3, yf, x] = 1.0 - frac
        mask[3, yf + 1, x] = frac
    return mask

def save_mask_coo(mask, output_path):
    C, H, W = mask.shape
    d = {'shape': np.array([C, H, W])}
    for i in range(C):
        ys, xs = np.nonzero(mask[i])
        d[f'ch{i}_y'] = ys.astype(np.int32)
        d[f'ch{i}_x'] = xs.astype(np.int32)
        d[f'ch{i}_v'] = mask[i, ys, xs].astype(np.float32)
    np.savez_compressed(output_path, **d)

for i, sid in enumerate(train_ids):
    out_path = f'{MASK_DIR}/{sid}.mask-coo.npz'
    if os.path.exists(out_path): continue
    try:
        create_mask_result = create_mask(f'{KAGGLE_DIR}/train/{sid}/{sid}.csv')
        save_mask_coo(create_mask_result, out_path)
    except Exception as e:
        print(f'  FAIL {sid}: {e}')
    if (i+1) % 200 == 0:
        print(f'  {i+1}/{len(train_ids)}')
print(f'  Done: {len(os.listdir(MASK_DIR))} masks')

# === Step 3: Copy ECG CSVs ===
print('\n=== Step 3: Copying ECG CSVs ===')
for sid in train_ids:
    src = f'{KAGGLE_DIR}/train/{sid}/{sid}.csv'
    dst = f'{ECG_DIR}/{sid}.csv'
    if not os.path.exists(dst): shutil.copy2(src, dst)
print(f'  Done: {len(os.listdir(ECG_DIR))} CSVs')

# === Step 4: Generate fold CSV ===
print('\n=== Step 4: Generating fold CSV ===')
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_map = {}
for fold_idx, (_, val_idx) in enumerate(kf.split(train_ids)):
    for idx in val_idx:
        fold_map[train_ids[idx]] = fold_idx

rows = []
for sid in train_ids:
    sig_len = train_df[train_df['id'] == sid].iloc[0]['sig_len']
    fold = fold_map[sid]
    if os.path.exists(f'{RECT_DIR}/{sid}-0001.rect.png'):
        rows.append({
            'id': sid, 'type_id': '0001', 'fold': fold,
            'sig_len': sig_len, 'is_synthesis': False,
            'image_path': f'data/rectified/{sid}-0001.rect.png',
        })
fold_csv = pd.DataFrame(rows)
fold_csv.to_csv(f'{OUT_DIR}/train_fold.csv', index=False)
print(f'  Done: {len(fold_csv)} rows')

# === Summary ===
print(f'\n=== SUMMARY ===')
print(f'Rectified images: {len(os.listdir(RECT_DIR))}')
print(f'Masks:            {len(os.listdir(MASK_DIR))}')
print(f'ECG CSVs:         {len(os.listdir(ECG_DIR))}')
rect_size = sum(os.path.getsize(os.path.join(RECT_DIR,f)) for f in os.listdir(RECT_DIR))
print(f'Total size:       ~{rect_size/1e9:.1f} GB')
print('DONE! Save this output as a dataset.')