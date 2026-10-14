"""
train_ensemble_local.py
=======================
Trains DenseNet-201 and ViT-Base on a 4GB VRAM GPU.
Uses Batch Size 8 and Mixed Precision (AMP) to prevent OOM errors.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import densenet201, DenseNet201_Weights, vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import time

from data.dataset import BrainTumorDataset, get_transforms
from data.config import DATASET_ROOT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH_SIZE = 8  # Safe for 4GB VRAM

# Glioma needs extra help because of poor recall
CLASS_WEIGHTS = torch.tensor([3.0, 1.0, 1.0, 1.5]).to(DEVICE) 

os.makedirs("checkpoints", exist_ok=True)

def train_model(model, model_name, train_loader, save_path):
    print(f"\n" + "="*50)
    print(f" TRAINING: {model_name} (For {EPOCHS} Epochs)")
    print("="*50)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Automatic Mixed Precision for 4GB VRAM
            with torch.amp.autocast('cuda'):
                logits = model(inputs)
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        print(f"--> Epoch {epoch+1} Summary | Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}% | Time: {time.time()-t0:.1f}s")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"  [+] Saved new best {model_name} weights to {save_path}")

    # Free up VRAM for the next model
    del model, optimizer, scaler
    torch.cuda.empty_cache()


def main():
    print(f"Detected Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print("\nLoading Training Dataset...")
    train_dataset = BrainTumorDataset(
        mode='train_val',
        transform=get_transforms(is_train=True),
        dataset_root=DATASET_ROOT
    )
    # 4 workers can sometimes crash on Windows laptops, using 2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print(f"Loaded {len(train_dataset)} images.\n")

    # 1. Train DenseNet-201
    densenet = densenet201(weights=DenseNet201_Weights.DEFAULT)
    densenet.classifier = nn.Linear(densenet.classifier.in_features, 4)
    train_model(densenet, "DenseNet-201", train_loader, "checkpoints/best_densenet_local.pth")

    # 2. Train ViT-Base (Much lighter than ViT-Large)
    vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    vit.heads.head = nn.Linear(vit.heads.head.in_features, 4)
    train_model(vit, "ViT-Base-16", train_loader, "checkpoints/best_vit_local.pth")

    print("\n" + "="*50)
    print(" ENSEMBLE TRAINING COMPLETE!")
    print(" Both models saved to the checkpoints directory.")
    print("="*50)

if __name__ == "__main__":
    main()
