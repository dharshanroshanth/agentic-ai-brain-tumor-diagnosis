import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from .config import CLASSES

def get_transforms(is_train=True, img_size=224):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def targeted_augmentation(inputs, labels, alpha=1.0, prob=0.5):
    """Mixup/Cutmix placeholder - using Mixup for simplicity"""
    if np.random.rand() > prob:
        return inputs, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size()[0]
    index = torch.randperm(batch_size).to(inputs.device)

    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    labels_a, labels_b = labels, labels[index]
    return mixed_inputs, labels_a, labels_b, lam

class BrainTumorDataset(Dataset):
    def __init__(self, mode='train_val', transform=None, dataset_root=''):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        search_dirs = []
        if mode == 'train_val':
            for d in ['Training', 'Train', 'Val']:
                path = os.path.join(dataset_root, d)
                if os.path.exists(path):
                    search_dirs.append(path)
        elif mode == 'test':
            for d in ['Testing', 'Test']:
                path = os.path.join(dataset_root, d)
                if os.path.exists(path):
                    search_dirs.append(path)
                    
        for search_dir in search_dirs:
            for class_name in os.listdir(search_dir):
                class_dir = os.path.join(search_dir, class_name)
                if not os.path.isdir(class_dir): continue
                
                # Normalize label
                label_str = class_name.lower().replace(' ', '_').replace('_tumor', '')
                if 'no' in label_str and 'tumor' in label_str: label_str = 'no_tumor'
                if label_str == 'notumor': label_str = 'no_tumor'
                
                if label_str in CLASSES:
                    label_idx = CLASSES.index(label_str)
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(class_dir, img_name))
                            self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback for corrupted images
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label
