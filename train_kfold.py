"""
NOTE: Run with: py -3.13 train_kfold.py
       Resume from fold 5: py -3.13 train_kfold.py --start-fold 5

Complete K-Fold Cross-Validation Training Script for the Perception Agent.
Runs EfficientNetV2-S with CutMix/MixUp augmentation.
Reports per-fold and aggregate metrics (accuracy, precision, recall, F1).
Saves best checkpoint per fold.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import BrainTumorDataset, get_transforms, targeted_augmentation
from data.config import CLASSES, DATASET_ROOT
from agents.perception import PerceptionAgent, safety_gate

# ============================================================
# HYPERPARAMETERS
# ============================================================
CONFIG = {
    "img_size":       224,
    "num_classes":    4,
    "batch_size":     32,
    "num_epochs":     30,
    "k_folds":        5,
    "lr":             1e-4,
    "weight_decay":   1e-4,
    "cutmix_prob":    0.5,
    "cutmix_alpha":   1.0,
    "early_stop_patience": 7,
    "checkpoint_dir": os.path.join(os.path.dirname(__file__), "checkpoints"),
    "results_dir":    os.path.join(os.path.dirname(__file__), "results"),
}

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs(CONFIG["results_dir"], exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory/1e9, 1)} GB")


def compute_class_weights(labels, num_classes=4):
    """Compute inverse-frequency class weights to handle imbalance."""
    from collections import Counter
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    weights = torch.FloatTensor(weights).to(DEVICE)
    print("\n[Class Weights for Imbalance Correction]")
    classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
    for i, (cls, w) in enumerate(zip(classes, weights)):
        print(f"  {cls:15s}: {w:.3f}  (n={counts.get(i,0)})")
    return weights


# ============================================================
# TRAIN EPOCH
# ============================================================
def train_epoch(model, loader, criterion, optimizer, scaler=None):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Targeted CutMix/MixUp
        inputs, labels_a, labels_b, lam = targeted_augmentation(
            inputs, labels,
            alpha=CONFIG["cutmix_alpha"],
            prob=CONFIG["cutmix_prob"]
        )

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast('cuda'):
                logits = model(inputs)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs)
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


# ============================================================
# EVAL EPOCH
# ============================================================
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Eval ", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================
def save_confusion_matrix(y_true, y_pred, fold, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - Fold {fold}")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"confusion_fold_{fold}.png"), dpi=120)
    plt.close(fig)
    print(f"  Saved confusion matrix to: results/confusion_fold_{fold}.png")


# ============================================================
# EFFICIENCY REPORTING
# ============================================================
def report_model_efficiency(model):
    """Count parameters and estimate FLOPs."""
    try:
        from thop import profile
        dummy = torch.randn(1, 3, CONFIG["img_size"], CONFIG["img_size"]).to(DEVICE)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        print(f"\n[Efficiency Report]")
        print(f"  Parameters : {params/1e6:.2f}M")
        print(f"  FLOPs      : {flops/1e9:.2f}G")
    except ImportError:
        params = sum(p.numel() for p in model.parameters())
        print(f"\n[Efficiency Report] Parameters: {params/1e6:.2f}M (install thop for FLOPs)")

    # Inference latency
    model.eval()
    dummy = torch.randn(1, 3, CONFIG["img_size"], CONFIG["img_size"]).to(DEVICE)
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(dummy)
        t0 = time.perf_counter()
        for _ in range(50):
            _ = model(dummy)
        latency_ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"  Inference latency: {latency_ms:.2f} ms/image (avg of 50 runs)")


# ============================================================
# MAIN K-FOLD TRAINING
# ============================================================
def run_kfold_training(start_fold: int = 1):
    print("\n" + "="*60)
    print("  BRAIN TUMOR DIAGNOSIS — EfficientNetV2 K-Fold Training")
    print("="*60)

    # Load combined train+val dataset (no transform yet — applied per-fold)
    full_dataset_raw = BrainTumorDataset(
        mode='train_val',
        transform=None,
        dataset_root=DATASET_ROOT
    )

    if len(full_dataset_raw) == 0:
        print("\nERROR: No images found! Check TRAIN_DIR path in data/config.py")
        return

    labels_arr = np.array(full_dataset_raw.labels)

    # Stratified K-Fold to ensure balanced class distribution per fold
    skf = StratifiedKFold(n_splits=CONFIG["k_folds"], shuffle=True, random_state=42)

    fold_results = {}

    if start_fold > 1:
        print(f"\n  [RESUME] Skipping folds 1–{start_fold-1}. Loading saved checkpoints...")
        # Re-evaluate already-completed folds from their saved checkpoints
        for prev_fold, (tr_idx, v_idx) in enumerate(skf.split(
                np.zeros(len(labels_arr)), labels_arr)):
            if prev_fold + 1 >= start_fold:
                break
            best_ckpt = os.path.join(CONFIG["results_dir"], f"best_fold_{prev_fold+1}.pth")
            if not os.path.exists(best_ckpt):
                print(f"  WARNING: Checkpoint for fold {prev_fold+1} not found — skipping.")
                continue
            val_dataset_prev = BrainTumorDataset(
                mode='train_val',
                transform=get_transforms(is_train=False, img_size=CONFIG["img_size"]),
                dataset_root=DATASET_ROOT
            )
            val_sub_prev = Subset(val_dataset_prev, v_idx)
            val_loader_prev = DataLoader(val_sub_prev, batch_size=CONFIG["batch_size"],
                                         shuffle=False, num_workers=2, pin_memory=True)
            model_prev = PerceptionAgent(num_classes=CONFIG["num_classes"], pretrained=False).to(DEVICE)
            model_prev.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
            criterion_prev = nn.CrossEntropyLoss()
            _, _, vl_preds_p, vl_labels_p = eval_epoch(model_prev, val_loader_prev, criterion_prev)
            acc_p   = accuracy_score(vl_labels_p, vl_preds_p)
            prec_p  = precision_score(vl_labels_p, vl_preds_p, average='weighted', zero_division=0)
            rec_p   = recall_score(vl_labels_p, vl_preds_p, average='weighted', zero_division=0)
            f1_p    = f1_score(vl_labels_p, vl_preds_p, average='weighted', zero_division=0)
            fold_results[prev_fold+1] = {
                "accuracy": acc_p, "precision": prec_p,
                "recall": rec_p, "f1": f1_p, "history": []
            }
            print(f"  Fold {prev_fold+1} (loaded): Acc={acc_p:.4f} | F1={f1_p:.4f}")
            del model_prev
            torch.cuda.empty_cache()
        print()

    for fold, (train_idx, val_idx) in enumerate(skf.split(
            np.zeros(len(labels_arr)), labels_arr)):

        if fold + 1 < start_fold:
            continue  # skip already-completed folds

        print(f"\n{'─'*50}")
        print(f"  FOLD {fold+1} / {CONFIG['k_folds']}")
        print(f"{'─'*50}")
        print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")

        # Apply transforms to subsets
        train_dataset = BrainTumorDataset(
            mode='train_val',
            transform=get_transforms(is_train=True, img_size=CONFIG["img_size"]),
            dataset_root=DATASET_ROOT
        )
        val_dataset = BrainTumorDataset(
            mode='train_val',
            transform=get_transforms(is_train=False, img_size=CONFIG["img_size"]),
            dataset_root=DATASET_ROOT
        )

        train_sub = Subset(train_dataset, train_idx)
        val_sub   = Subset(val_dataset, val_idx)

        train_loader = DataLoader(train_sub, batch_size=CONFIG["batch_size"],
                                  shuffle=True, num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_sub, batch_size=CONFIG["batch_size"],
                                  shuffle=False, num_workers=2, pin_memory=True)

        # Compute class weights from this fold's training labels
        fold_train_labels = [full_dataset_raw.labels[i] for i in train_idx]
        class_weights = compute_class_weights(fold_train_labels, CONFIG["num_classes"])

        # Model, loss, optimizer
        model     = PerceptionAgent(num_classes=CONFIG["num_classes"], pretrained=True).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                                weight_decay=CONFIG["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6
        )

        # AMP scaler for faster GPU training (torch.amp API for PyTorch 2.6+)
        scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

        best_val_acc  = 0.0
        best_ckpt     = os.path.join(CONFIG["checkpoint_dir"], f"best_fold{fold+1}.pth")
        patience_cnt  = 0
        history       = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

        for epoch in range(CONFIG["num_epochs"]):
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
            vl_loss, vl_acc, vl_preds, vl_labels = eval_epoch(model, val_loader, criterion)
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(vl_acc)

            print(f"  Ep [{epoch+1:02d}/{CONFIG['num_epochs']}] "
                  f"TrainLoss={tr_loss:.4f} Acc={tr_acc:.4f} | "
                  f"ValLoss={vl_loss:.4f} Acc={vl_acc:.4f}", end="")

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                torch.save(model.state_dict(), best_ckpt)
                patience_cnt = 0
                print(" ✓ SAVED")
            else:
                patience_cnt += 1
                print()

            if patience_cnt >= CONFIG["early_stop_patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Final evaluation with best checkpoint
        model.load_state_dict(torch.load(best_ckpt))
        _, _, vl_preds, vl_labels = eval_epoch(model, val_loader, criterion)

        precision = precision_score(vl_labels, vl_preds, average='weighted', zero_division=0)
        recall    = recall_score(vl_labels, vl_preds, average='weighted', zero_division=0)
        f1        = f1_score(vl_labels, vl_preds, average='weighted', zero_division=0)

        print(f"\n  Fold {fold+1} Results:")
        print(f"    Best Val Accuracy : {best_val_acc:.4f}")
        print(f"    Precision (w-avg) : {precision:.4f}")
        print(f"    Recall (w-avg)    : {recall:.4f}")
        print(f"    F1 Score (w-avg)  : {f1:.4f}")
        print(f"\n  Per-Class Report:\n")
        print(classification_report(vl_labels, vl_preds, target_names=CLASSES))

        save_confusion_matrix(vl_labels, vl_preds, fold+1, CONFIG["results_dir"])

        fold_results[fold+1] = {
            "accuracy":  best_val_acc,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "history":   history
        }

    # ── AGGREGATE RESULTS ──────────────────────────────────
    print("\n" + "="*60)
    print("  K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*60)

    accs  = [fold_results[f]["accuracy"]  for f in fold_results]
    precs = [fold_results[f]["precision"] for f in fold_results]
    recs  = [fold_results[f]["recall"]    for f in fold_results]
    f1s   = [fold_results[f]["f1"]        for f in fold_results]

    print(f"  Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Precision : {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"  Recall    : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"  F1 Score  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # Save results
    results_path = os.path.join(CONFIG["results_dir"], "kfold_results.txt")
    with open(results_path, "w") as f:
        f.write("K-Fold Cross-Validation Results\n")
        f.write("="*50 + "\n")
        for fold_num, res in fold_results.items():
            f.write(f"Fold {fold_num}: Acc={res['accuracy']:.4f} "
                    f"P={res['precision']:.4f} R={res['recall']:.4f} "
                    f"F1={res['f1']:.4f}\n")
        f.write("-"*50 + "\n")
        f.write(f"Mean Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}\n")
        f.write(f"Mean Precision : {np.mean(precs):.4f} ± {np.std(precs):.4f}\n")
        f.write(f"Mean Recall    : {np.mean(recs):.4f} ± {np.std(recs):.4f}\n")
        f.write(f"Mean F1        : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n")
    print(f"\n  Results saved to: {results_path}")

    # Report model efficiency
    report_model_efficiency(PerceptionAgent(num_classes=4, pretrained=False).to(DEVICE))

    return fold_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="K-Fold Training")
    parser.add_argument("--start-fold", type=int, default=1,
                        help="Start from this fold number (1-indexed). Folds before this are loaded from saved checkpoints.")
    args = parser.parse_args()
    run_kfold_training(start_fold=args.start_fold)
