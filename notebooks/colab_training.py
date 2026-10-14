# Brain Tumor Diagnosis Assistant - Google Colab Notebook
# Run this notebook on Google Colab with a T4 GPU

# ============================================================
# CELL 1: Clone / Upload Project
# ============================================================
# Option A: Clone from GitHub (recommended)
# !git clone https://github.com/YOUR_USERNAME/brain-tumor-agent.git
# %cd brain-tumor-agent

# Option B: Upload zip
# from google.colab import files
# files.upload()  # upload project.zip
# !unzip project.zip

# ============================================================
# CELL 2: Install Dependencies
# ============================================================
SETUP = """
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install grad-cam shap langchain langchain-openai langchain-community faiss-cpu
!pip install scikit-learn matplotlib seaborn tqdm thop
"""

# ============================================================
# CELL 3: Upload Dataset (or mount Drive)
# ============================================================
DATASET_SETUP = """
from google.colab import drive
drive.mount('/content/drive')

import os
# Update this path to where you store the dataset in Drive
DATASET_ROOT = '/content/drive/MyDrive/brain_tumor_dataset'

# OR: Kaggle download
# !pip install kaggle
# from google.colab import files
# files.upload()  # upload kaggle.json
# !mkdir ~/.kaggle && mv kaggle.json ~/.kaggle/
# !kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
# !unzip brain-tumor-mri-dataset.zip -d dataset/
"""

# ============================================================
# CELL 4: Update dataset root in config.py
# ============================================================
UPDATE_CONFIG = """
# Update DATASET_ROOT in data/config.py to point to Colab path
import re

config_path = 'data/config.py'
with open(config_path, 'r') as f:
    content = f.read()

# Replace the Windows path with the Colab path
content = re.sub(
    r"DATASET_ROOT = r.*",
    f'DATASET_ROOT = r"{DATASET_ROOT}"',
    content
)

with open(config_path, 'w') as f:
    f.write(content)

print("Config updated!")
"""

# ============================================================
# CELL 5: Verify Dataset Loading
# ============================================================
VERIFY = """
import sys
sys.path.insert(0, '.')
from data.dataset import BrainTumorDataset

print("=== Verifying Dataset Splits ===")
for mode in ['train', 'val', 'test', 'train_val']:
    ds = BrainTumorDataset(mode=mode)
    print(f"  {mode}: {len(ds)} images")
"""

# ============================================================
# CELL 6: Start K-Fold Training
# ============================================================
TRAIN = """
# Enable GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Run training
from train_kfold import run_kfold_training
fold_results = run_kfold_training()
"""

# ============================================================
# CELL 7: External Validation
# ============================================================
EVALUATE = """
from evaluate_external import run_external_validation
run_external_validation()
"""

# ============================================================
# CELL 8: Test the Full Pipeline on a Sample Image
# ============================================================
PIPELINE_TEST = """
import os
import glob
from pipeline import DiagnosticPipeline

# Find a test image
test_images = glob.glob('dataset/Testing/glioma/*.jpg')
if test_images:
    test_img = test_images[0]
    pipeline = DiagnosticPipeline()
    report = pipeline.run_case("CASE-001", test_img)
    print(report)
else:
    print("No test images found. Check dataset path.")
"""

print("Colab notebook template loaded.")
print("Copy each CELL_* variable into a Colab code cell to run sequentially.")
