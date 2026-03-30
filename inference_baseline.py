# Cell 1: Setup & Install
!pip install connected-components-3d segmentation-models-pytorch --no-index --find-links=file:///kaggle/input/datasets/tylerde/my-pip-packages/

# Cell 2: Imports & Config
import cc3d
import cv2
import pandas as pd
import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import shutil
import copy
import multiprocessing as mp
import pickle
import os
import sys

print('import ok!!!')

FLOAT_TYPE = torch.float16

# === PATHS ===
KAGGLE_DIR   = '/kaggle/input/competitions/physionet-ecg-image-digitization'
HENGCK23_DIR = '/kaggle/input/datasets/hengck23/hengck23-demo-submit-physionet'
SOMEYA_DIR   = '/kaggle/input/datasets/takashisomeya/physionet-final-submission-models'
WEIGHT_DIR   = f'{HENGCK23_DIR}/weight'
OUT_DIR      = '/kaggle/working/outputs'

# Add to sys.path
sys.path.append(HENGCK23_DIR)
sys.path.append(SOMEYA_DIR)

# === Read test metadata ===
valid_df = pd.read_csv(f'{KAGGLE_DIR}/test.csv')
valid_df['id'] = valid_df['id'].astype(str)
valid_id = valid_df['id'].unique().tolist()

def read_image(sample_id):
    return cv2.imread(f'{KAGGLE_DIR}/test/{sample_id}.png', cv2.IMREAD_COLOR_RGB)

def read_sampling_length(sample_id):
    d = valid_df[
        (valid_df['id'] == sample_id) & (valid_df['lead'] == 'II')
    ].iloc[0]
    return d.number_of_rows

print(f'valid_id: {len(valid_id)}')
print(f'\t{valid_id[:3]} ...')
print('setting ok!!!\n')

# Cell 3: Stage 0 — Keypoint Detection & Normalization
print('*** STARTING STAGE0 ***')

from stage0_model import Net as Stage0Net
from stage0_common import *

os.makedirs(f'{OUT_DIR}/normalised', exist_ok=True)

def run_stage0(gpu_id=0, assigned_ids=None, fail_id_file=None):
    device = f'cuda:{gpu_id}'
    if assigned_ids is None:
        assigned_ids = valid_id

    local_fail_id = []

    stage0_net = Stage0Net(pretrained=False)
    stage0_net = load_net(stage0_net, f'{WEIGHT_DIR}/stage0-last.checkpoint.pth')
    stage0_net.to(device)

    start_timer = timer()
    for n, sample_id in enumerate(assigned_ids):
        timestamp = time_to_str(timer() - start_timer, 'sec')
        print(f'\r\t [GPU{gpu_id}] {n:4d}/{len(assigned_ids)} {sample_id}', timestamp, end='', flush=True)

        image = read_image(sample_id)
        batch = image_to_batch(image)

        with torch.amp.autocast('cuda', dtype=FLOAT_TYPE):
            with torch.no_grad():
                output = stage0_net(batch)
                try:
                    rotated, keypoint = output_to_predict(image, batch, output)
                    normalised, keypoint, homo = normalise_by_homography(rotated, keypoint)
                    cv2.imwrite(f'{OUT_DIR}/normalised/{sample_id}.norm.png', cv2.cvtColor(normalised, cv2.COLOR_RGB2BGR))
                    np.save(f'{OUT_DIR}/normalised/{sample_id}.homo.npy', homo)
                except:
                    local_fail_id.append(sample_id)

        torch.cuda.empty_cache()

        if n < 3 and gpu_id == 0:
            overlay = draw_results_stage0(rotated, keypoint)
            print(f'\ndemo results for stage0: {sample_id}')
            plt.imshow(image); plt.show()
            plt.imshow(overlay); plt.show()
            plt.imshow(normalised); plt.show()

    print(f'\n[GPU{gpu_id}] Stage0 completed. Failed: {len(local_fail_id)}')

    if fail_id_file:
        with open(fail_id_file, 'wb') as f:
            pickle.dump(local_fail_id, f)

    return local_fail_id


def run_stage0_parallel():
    print('*** STARTING STAGE0 (2GPU PARALLEL) ***')
    n_gpus = torch.cuda.device_count()

    if n_gpus < 2:
        print(f'Only {n_gpus} GPU(s) available, running single GPU')
        return run_stage0(gpu_id=0)

    mid_idx = len(valid_id) // 2
    ids_gpu0, ids_gpu1 = valid_id[:mid_idx], valid_id[mid_idx:]
    print(f'GPU0: {len(ids_gpu0)} | GPU1: {len(ids_gpu1)}')

    fail_file_0 = f'{OUT_DIR}/fail_stage0_gpu0.pkl'
    fail_file_1 = f'{OUT_DIR}/fail_stage0_gpu1.pkl'

    p0 = mp.Process(target=run_stage0, args=(0, ids_gpu0, fail_file_0))
    p1 = mp.Process(target=run_stage0, args=(1, ids_gpu1, fail_file_1))
    p0.start(); p1.start()
    p0.join();  p1.join()

    fail_id = []
    for ff in [fail_file_0, fail_file_1]:
        if os.path.exists(ff):
            with open(ff, 'rb') as f:
                fail_id.extend(pickle.load(f))

    print(f'FAIL_ID (Stage0): {fail_id}')
    return fail_id


FAIL_ID_STAGE0 = run_stage0_parallel()
print('Stage0 done!\n')

# Cell 4: Stage 1 — Grid Detection & Rectification
print('*** STARTING STAGE1 ***')

from stage1_model import Net as Stage1Net
from stage1_common import *

os.makedirs(f'{OUT_DIR}/rectified', exist_ok=True)

def run_stage1(gpu_id=0, assigned_ids=None, prev_fail_ids=None, fail_id_file=None):
    device = f'cuda:{gpu_id}'
    if assigned_ids is None:
        assigned_ids = valid_id
    if prev_fail_ids is None:
        prev_fail_ids = []

    local_fail_id = []

    stage1_net = Stage1Net(pretrained=False)
    stage1_net = load_net(stage1_net, f'{WEIGHT_DIR}/stage1-last.checkpoint.pth')
    stage1_net.to(device)

    start_timer = timer()
    for n, sample_id in enumerate(assigned_ids):
        timestamp = time_to_str(timer() - start_timer, 'sec')
        print(f'\r\t [GPU{gpu_id}] {n:4d}/{len(assigned_ids)} {sample_id}', timestamp, end='', flush=True)

        if sample_id in prev_fail_ids:
            continue

        image = cv2.imread(f'{OUT_DIR}/normalised/{sample_id}.norm.png', cv2.IMREAD_COLOR_RGB)
        batch = {
            'image': torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0),
        }

        with torch.amp.autocast('cuda', dtype=FLOAT_TYPE):
            with torch.no_grad():
                output = stage1_net(batch)
                try:
                    gridpoint_xy, more = output_to_predict(image, batch, output)
                    rectified = rectify_image(image, gridpoint_xy)
                    cv2.imwrite(f'{OUT_DIR}/rectified/{sample_id}.rect.png', cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR))
                    np.save(f'{OUT_DIR}/rectified/{sample_id}.gridpoint_xy.npy', gridpoint_xy)
                except:
                    local_fail_id.append(sample_id)

        torch.cuda.empty_cache()

        if n < 3 and gpu_id == 0:
            overlay = draw_mapping(image, gridpoint_xy)
            ghfiltered, gvfiltered = draw_results_stage1(more)
            print(f'\ndemo results for stage1: {sample_id}')
            plt.imshow(overlay); plt.show()
            plt.imshow(gvfiltered); plt.show()
            plt.imshow(ghfiltered); plt.show()
            plt.imshow(rectified); plt.show()

    print(f'\n[GPU{gpu_id}] Stage1 completed. Failed: {len(local_fail_id)}')

    if fail_id_file:
        with open(fail_id_file, 'wb') as f:
            pickle.dump(local_fail_id, f)

    return local_fail_id


def run_stage1_parallel(prev_fail_ids=None):
    print('*** STARTING STAGE1 (2GPU PARALLEL) ***')
    n_gpus = torch.cuda.device_count()

    if n_gpus < 2:
        print(f'Only {n_gpus} GPU(s) available, running single GPU')
        return run_stage1(gpu_id=0, prev_fail_ids=prev_fail_ids)

    mid_idx = len(valid_id) // 2
    ids_gpu0, ids_gpu1 = valid_id[:mid_idx], valid_id[mid_idx:]
    print(f'GPU0: {len(ids_gpu0)} | GPU1: {len(ids_gpu1)}')

    fail_file_0 = f'{OUT_DIR}/fail_stage1_gpu0.pkl'
    fail_file_1 = f'{OUT_DIR}/fail_stage1_gpu1.pkl'

    p0 = mp.Process(target=run_stage1, args=(0, ids_gpu0, prev_fail_ids, fail_file_0))
    p1 = mp.Process(target=run_stage1, args=(1, ids_gpu1, prev_fail_ids, fail_file_1))
    p0.start(); p1.start()
    p0.join();  p1.join()

    fail_id = []
    for ff in [fail_file_0, fail_file_1]:
        if os.path.exists(ff):
            with open(ff, 'rb') as f:
                fail_id.extend(pickle.load(f))
    if prev_fail_ids:
        fail_id.extend(prev_fail_ids)

    print(f'FAIL_ID (Stage1): {fail_id}')
    return fail_id


FAIL_ID_STAGE1 = run_stage1_parallel(prev_fail_ids=FAIL_ID_STAGE0)
print('Stage1 done!\n')

# Cell 5: Stage 2 — Segmentation & Signal Extraction
print('*** STARTING STAGE2 ***')

from stage2_smp_model import Net as WholeModel
from stage2_lead_model import Net as LeadModel
from stage2_model import prob_to_series_by_max
from stage2_common import *

os.makedirs(f'{OUT_DIR}/digitalised', exist_ok=True)

# === Stage 2 Constants ===
WINDOW_SIZE = 240
OFFSET = 416
IGNORE_EDGE = 8
x_scale = 5000 / (2080 - 118)
add_x = 1
y_scale = 1
IMG_H, IMG_W = int(1700 * y_scale), int(2200 * x_scale) + add_x

tta = [0, 2]  # no flip, horizontal flip

x0, x1 = 0, 5600
y0, y1 = 0, 1696
zero_mv = [703.5, 987.5, 1271.5, 1531.5]
zero_mv_trimed = [pos - OFFSET for pos in zero_mv]
zero_mv_croped = [WINDOW_SIZE + 0.5 for _ in range(4)]
mv_to_pixel = 79.0
t0 = int(118 * x_scale) + add_x
t1 = int(2080 * x_scale) + add_x

# Pre-compute ensemble regions
height_after_trimed = y1 - OFFSET
ens_regions = []
for zmv in zero_mv_trimed:
    trim_upper = int(zmv) - WINDOW_SIZE
    trim_lower = int(zmv) + WINDOW_SIZE
    lead_upper = IGNORE_EDGE
    lead_lower = -IGNORE_EDGE
    if trim_lower > height_after_trimed:
        lead_lower = (trim_lower - height_after_trimed + IGNORE_EDGE) * -1
        trim_lower = height_after_trimed
    trim_upper += IGNORE_EDGE
    trim_lower -= IGNORE_EDGE
    ens_regions.append([trim_upper, trim_lower, lead_upper, lead_lower])

print(f'IMG: ({IMG_H}, {IMG_W}), timespan: [{t0}:{t1}]')
print(f'ens_regions: {ens_regions}')


def pixel_to_series_exp(pixel, zero_mv, length):
    _, H, W = pixel.shape
    eps = 1e-8
    y_idx = np.arange(H, dtype=np.float32)[:, None]

    series = []
    for j in [0, 1, 2, 3]:
        p = pixel[j]
        denom = p.sum(axis=0)
        y_exp = (p * y_idx).sum(axis=0) / (denom + eps)
        series.append(y_exp)
    series = np.stack(series).astype(np.float32)

    if length is not None and length != W:
        resampled_series = []
        for s in series:
            rs = signal.resample(s, length).astype(np.float32)
            resampled_series.append(rs)
        series = np.stack(resampled_series)

    return series


def get_whole_model(encoder_name, weight_path, device):
    model = WholeModel(
        encoder_name=encoder_name,
        encoder_weights=None,
        decoder_name="unet",
        use_coord_conv=True,
        pretrained=False
    )
    state_dict = torch.load(weight_path, map_location='cpu')
    print(model.load_state_dict(state_dict, strict=False))
    model.to(device)
    model.eval()
    model.output_type = ['infer']
    return model


def get_lead_model(encoder_name, weight_path, fusion_type, device):
    model = LeadModel(
        encoder_name=encoder_name,
        encoder_weights=None,
        fusion_type=fusion_type,
    )
    state_dict = torch.load(weight_path, map_location='cpu')
    print(model.load_state_dict(state_dict, strict=False))
    model.to(device)
    model.eval()
    model.output_type = ['infer']
    return model


def read_images(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    trim_image = image.copy()[OFFSET:y1, x0:x1]

    # Make lead images
    image = image[y0:y1, x0:x1]
    H, W, _ = image.shape
    lead_images = []
    for i, zmv in enumerate(zero_mv):
        h0, h1 = int(zmv) - WINDOW_SIZE, int(zmv) + WINDOW_SIZE
        src_h0 = max(0, h0)
        src_h1 = min(H, h1)
        dst_h0 = src_h0 - h0
        dst_h1 = dst_h0 + (src_h1 - src_h0)

        lead_img = np.zeros((WINDOW_SIZE * 2, W, 3))
        lead_img[dst_h0:dst_h1, :, :] = image[src_h0:src_h1, :, :]
        lead_images.append(lead_img)

    lead_images = np.stack(lead_images)  # (4, H, W, 3)
    return trim_image, lead_images


def run_stage2(gpu_id=0, assigned_ids=None, prev_fail_ids=None, fail_id_file=None):
    device = f'cuda:{gpu_id}'
    if assigned_ids is None:
        assigned_ids = valid_id
    if prev_fail_ids is None:
        prev_fail_ids = []

    local_fail_id = []

    # Load whole models
    whole_models = [
        get_whole_model("tu-timm/tf_efficientnet_b7.ns_jft_in1k",
                        f"{SOMEYA_DIR}/whole_b7_lb22.93.pth", device),
        get_whole_model("tu-timm/tf_efficientnetv2_l.in21k",
                        f"{SOMEYA_DIR}/whole_v2_l_lb22.60.pth", device),
    ]

    # Load lead/series models
    lead_models = [
        get_lead_model("tu-timm/tf_efficientnet_b6.ns_jft_in1k",
                       f"{SOMEYA_DIR}/series_b6_shared_conv2d_lb23.10.pth", "shared_conv2d", device),
        get_lead_model("tu-timm/tf_efficientnet_b6.ns_jft_in1k",
                       f"{SOMEYA_DIR}/series_b6_shared_conv2d_lb23.00.pth", "shared_conv2d", device),
        get_lead_model("tu-timm/tf_efficientnetv2_l.in21k",
                       f"{SOMEYA_DIR}/series_v2_l_conv3d_lb22.92.pth", "conv3d", device),
        get_lead_model("tu-timm/tf_efficientnetv2_l.in21k",
                       f"{SOMEYA_DIR}/series_v2_l_conv2d_lb22.85.pth", "conv2d", device),
    ]

    start_timer = timer()
    for n, sample_id in enumerate(assigned_ids):
        timestamp = time_to_str(timer() - start_timer, 'sec')
        print(f'\r\t [GPU{gpu_id}] {n:4d}/{len(assigned_ids)} {sample_id}', timestamp, end='', flush=True)

        if sample_id in prev_fail_ids:
            continue

        length = read_sampling_length(sample_id)
        trim_image, lead_images = read_images(f'{OUT_DIR}/rectified/{sample_id}.rect.png')

        pixel_ens = np.zeros((4, trim_image.shape[0], trim_image.shape[1])) * 1.0

        # --- Whole models ---
        batch = {
            'image': torch.from_numpy(
                np.ascontiguousarray(trim_image.transpose(2, 0, 1))
            ).unsqueeze(0),
        }
        batch_tta = {
            'image': torch.from_numpy(
                np.ascontiguousarray(np.fliplr(trim_image).copy().transpose(2, 0, 1))
            ).unsqueeze(0),
        }

        with torch.amp.autocast('cuda', dtype=FLOAT_TYPE):
            with torch.no_grad():
                for model in whole_models:
                    for flip in tta:
                        if flip:
                            output = model(batch_tta)
                            pixel = output['pixel'].float().data.cpu().numpy()[0]
                            pixel = np.flip(pixel, axis=flip)
                        else:
                            output = model(batch)
                            pixel = output['pixel'].float().data.cpu().numpy()[0]
                        pixel_ens += pixel

        # --- Lead/series models ---
        lead_tensor = torch.from_numpy(
            lead_images.transpose(0, 3, 1, 2)
        ).contiguous()  # (4, 3, H, W)
        batch = {'image': lead_tensor.unsqueeze(0)}
        batch_tta = {'image': torch.flip(lead_tensor, dims=[3]).unsqueeze(0)}

        with torch.amp.autocast('cuda', dtype=FLOAT_TYPE):
            with torch.no_grad():
                for model in lead_models:
                    for flip in tta:
                        if flip:
                            output = model(batch_tta)
                            pixel = output['pixel'].float().data.cpu().numpy()[0].squeeze(1)
                            pixel = np.flip(pixel, axis=flip)
                        else:
                            output = model(batch)
                            pixel = output['pixel'].float().data.cpu().numpy()[0].squeeze(1)

                        for i in range(4):
                            trim_upper, trim_lower, lead_upper, lead_lower = ens_regions[i]
                            pixel_ens[i][trim_upper:trim_lower] += pixel[i][lead_upper:lead_lower]

        # --- Weighted average ---
        ens_weight = np.ones((trim_image.shape[0], trim_image.shape[1])) * len(whole_models) * len(tta)
        for i in range(4):
            trim_upper, trim_lower, _, _ = ens_regions[i]
            ens_weight[trim_upper:trim_lower] += len(lead_models) * len(tta)
        pixel_ens /= ens_weight

        # --- Convert pixel → time series ---
        try:
            series_in_pixel = pixel_to_series_exp(pixel_ens[..., t0:t1], zero_mv_trimed, length)
            series = (np.array(zero_mv_trimed).reshape(4, 1) - series_in_pixel) / mv_to_pixel
            np.save(f'{OUT_DIR}/digitalised/{sample_id}.series.npy', series)
        except:
            local_fail_id.append(sample_id)

        if n < 3 and gpu_id == 0:
            print()
            print(f"check max intensity: {np.max(pixel_ens)}")
            overlay = draw_lead_pixel(trim_image, pixel_ens)
            plt.imshow(overlay); plt.show()
            t = np.arange(len(series[0]))
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            for j in range(4):
                axes[j].plot(t, series[j], alpha=1.0, color='blue', linewidth=1, label='predict')
                axes[j].legend()
            plt.tight_layout(); plt.show()

    print(f'\n[GPU{gpu_id}] Stage2 completed. Failed: {len(local_fail_id)}')

    if fail_id_file:
        with open(fail_id_file, 'wb') as f:
            pickle.dump(local_fail_id, f)

    return local_fail_id


def run_stage2_parallel(prev_fail_ids=None):
    print('*** STARTING STAGE2 (2GPU PARALLEL) ***')
    n_gpus = torch.cuda.device_count()

    if n_gpus < 2:
        print(f'Only {n_gpus} GPU(s) available, running single GPU')
        return run_stage2(gpu_id=0, prev_fail_ids=prev_fail_ids)

    mid_idx = len(valid_id) // 2
    ids_gpu0, ids_gpu1 = valid_id[:mid_idx], valid_id[mid_idx:]
    print(f'GPU0: {len(ids_gpu0)} | GPU1: {len(ids_gpu1)}')

    fail_file_0 = f'{OUT_DIR}/fail_stage2_gpu0.pkl'
    fail_file_1 = f'{OUT_DIR}/fail_stage2_gpu1.pkl'

    p0 = mp.Process(target=run_stage2, args=(0, ids_gpu0, prev_fail_ids, fail_file_0))
    p1 = mp.Process(target=run_stage2, args=(1, ids_gpu1, prev_fail_ids, fail_file_1))
    p0.start(); p1.start()
    p0.join();  p1.join()

    fail_id = []
    for ff in [fail_file_0, fail_file_1]:
        if os.path.exists(ff):
            with open(ff, 'rb') as f:
                fail_id.extend(pickle.load(f))
    if prev_fail_ids:
        fail_id.extend(prev_fail_ids)

    print(f'FAIL_ID (Stage2): {fail_id}')
    return fail_id


FAIL_ID_STAGE2 = run_stage2_parallel(prev_fail_ids=FAIL_ID_STAGE1)
print('Stage2 done!\n')

# Cell 6: Build Submission
def make_submission():
    print('===========================================')
    print('Making submission CSV ...')

    submit_df = []
    gb = valid_df.groupby('id')

    for i, (sample_id, df) in enumerate(gb):
        try:
            series = np.load(f'{OUT_DIR}/digitalised/{sample_id}.series.npy')

            series_by_lead = {}
            for l in range(3):
                lead = [
                    ['I',   'aVR', 'V1', 'V4'],
                    ['II',  'aVL', 'V2', 'V5'],
                    ['III', 'aVF', 'V3', 'V6'],
                ][l]

                length = [
                    df[df['lead'] == lead[j]].iloc[0].number_of_rows
                    for j in range(4)
                ]
                if lead[0] == 'II':
                    length[0] = length[0] - sum(length[1:])

                index = np.cumsum(length)[:-1]
                split = np.split(series[l], index)
                for (k, s) in zip(lead, split):
                    series_by_lead[k] = s

            series_by_lead['II'] = series[3]

        except:
            series_by_lead = {}
            for j, d in df.iterrows():
                series_by_lead[d.lead] = np.zeros(d.number_of_rows)

        for j, d in df.iterrows():
            # Safety pad: ensure correct length
            series_by_lead[d.lead] = np.concatenate([
                series_by_lead[d.lead], np.zeros_like(series_by_lead[d.lead])
            ])[:d.number_of_rows]
            assert len(series_by_lead[d.lead]) == d.number_of_rows

            print(f'\r\t {i} {sample_id} : {d.lead}', end='', flush=True)

            row_id = [f'{sample_id}_{i}_{d.lead}' for i in range(d.number_of_rows)]
            this_df = pd.DataFrame({
                'id': row_id,
                'value': series_by_lead[d.lead].astype(np.float32),
            })
            submit_df.append(this_df)

    print('')
    submit_df = pd.concat(submit_df, axis=0, ignore_index=True, sort=False, copy=False)
    print(submit_df)
    print(f'\nTotal rows: {len(submit_df)}')
    submit_df.to_csv('submission.csv', index=False)
    print('Saved submission.csv')


make_submission()

# ## Cell 7: Cleanup

# %%
shutil.rmtree(OUT_DIR)
print('Cleanup done!')
print('Ready to submit submission.csv')