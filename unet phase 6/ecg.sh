#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32gb
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -J ecg
#SBATCH -o output_%j.txt
#SBATCH -e error_%j.txt

module load miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate myConda

cd ~/ecg-training-all-samples

echo "=== GPU ==="
nvidia-smi | head -5
python -c "import torch; print(f'PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name()}')"

echo "=== Fix folds ==="
python -c "
import pandas as pd, numpy as np
df = pd.read_csv('data/train_fold.csv', dtype={'id': str, 'type_id': str})
np.random.seed(42)
ids = df['id'].unique()
np.random.shuffle(ids)
split = int(len(ids) * 0.8)
val_ids = set(ids[split:])
df['fold'] = df['id'].apply(lambda x: 0 if x in val_ids else 1)
df.to_csv('data/train_fold.csv', index=False)
print(f'Train: {len(df[df[\"fold\"] != 0])}, Val: {len(df[df[\"fold\"] == 0])}')
"

echo "=== Training ==="
python train.py --resume checkpoints/last_fold0.pth