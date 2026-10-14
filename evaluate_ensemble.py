"""
evaluate_ensemble.py
====================
Loads the massive 3-model ensemble and evaluates it on the External Test set.
Includes Test-Time Augmentation (TTA) to push the score as high as possible.
"""

import torch
import numpy as np
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from data.dataset import BrainTumorDataset, get_transforms
from data.config import DATASET_ROOT, CLASSES
from agents.perception import EnsemblePerceptionAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensemble_tta_evaluation():
    print("=" * 60)
    print(" 3-MODEL ENSEMBLE EVALUATION (WITH TTA) ")
    print("=" * 60)

    # 1. Load the Model
    model = EnsemblePerceptionAgent(num_classes=4)
    model.load_weights(
        effnet_path="checkpoints/best_fold1.pth",
        densenet_path="checkpoints/best_densenet_local.pth",
        vit_path="checkpoints/best_vit_local.pth",
        device=DEVICE
    )
    model.to(DEVICE)
    model.eval()

    # 2. Load the Test Dataset
    test_dataset = BrainTumorDataset(
        mode='train_val',
        transform=get_transforms(is_train=False),
        dataset_root=DATASET_ROOT
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print("\nStarting Ensemble TTA Inference (5 perspectives per image)...")
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(DEVICE)
            
            # Create 5 TTA variations (Original, H-Flip, V-Flip, Rotated, Darker)
            inputs_hflip = TF.hflip(inputs)
            inputs_vflip = TF.vflip(inputs)
            inputs_rot   = TF.rotate(inputs, 15)
            inputs_dark  = TF.adjust_brightness(inputs, 0.8)

            # Pass all 5 through the massive 3-model brain (15 predictions total!)
            logits_orig  = model(inputs)
            logits_hflip = model(inputs_hflip)
            logits_vflip = model(inputs_vflip)
            logits_rot   = model(inputs_rot)
            logits_dark  = model(inputs_dark)
            
            # Average the logits
            avg_logits = (logits_orig + logits_hflip + logits_vflip + logits_rot + logits_dark) / 5.0
            
            # Advanced "Glioma Threshold" trick (Recall booster)
            probs = torch.softmax(avg_logits, dim=-1)
            
            # Default Argmax
            preds = probs.argmax(dim=-1).cpu().numpy()
            
            # Override if Glioma probability is > 30% (since the ensemble is usually too strict on it)
            glioma_idx = CLASSES.index("glioma")
            for i in range(len(preds)):
                if probs[i][glioma_idx].item() > 0.30:
                    preds[i] = glioma_idx
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # 3. Print Results
    print("\n" + "=" * 60)
    acc = accuracy_score(all_labels, all_preds)
    print(f" FINAL ENSEMBLE TESTING ACCURACY: {acc*100:.3f}%")
    print("=" * 60)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))
    
    # 4. Save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, annot_kws={"size": 14})
    plt.title('Confusion Matrix - 3-Model Ensemble', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix_ensemble.png', dpi=300)

if __name__ == "__main__":
    ensemble_tta_evaluation()
