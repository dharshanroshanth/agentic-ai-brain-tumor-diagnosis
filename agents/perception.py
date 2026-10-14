import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    densenet201, DenseNet201_Weights,
    vit_b_16, ViT_B_16_Weights
)

class PerceptionAgent(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_v2_s(weights=weights)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class DenseNetAgent(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        super().__init__()
        weights = DenseNet201_Weights.DEFAULT if pretrained else None
        self.backbone = densenet201(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class ViTAgent(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class EnsemblePerceptionAgent(nn.Module):
    """
    Combines EfficientNet, DenseNet, and ViT. 
    Averages their prediction probabilities.
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.effnet = PerceptionAgent(num_classes, pretrained=False)
        self.densenet = DenseNetAgent(num_classes, pretrained=False)
        self.vit = ViTAgent(num_classes, pretrained=False)
        
        self.active_models = []

    def load_weights(self, effnet_path, densenet_path, vit_path, device='cpu'):
        self.effnet_loaded   = False
        self.densenet_loaded = False
        self.vit_loaded      = False

        # 1. EfficientNet
        try:
            raw_state = torch.load(effnet_path, map_location=device)
            new_state = {}
            for k, v in raw_state.items():
                if not k.startswith("backbone."):
                    new_state["backbone." + k] = v
                else:
                    new_state[k] = v
            self.effnet.load_state_dict(new_state, strict=False)
            self.effnet_loaded = True
            self.active_models.append(self.effnet)
            print("[Ensemble] EfficientNet loaded successfully.")
        except Exception as e:
            print(f"[Ensemble] EfficientNet skipped: {e}")

        # 2. DenseNet
        try:
            self.densenet.backbone.load_state_dict(torch.load(densenet_path, map_location=device))
            self.densenet_loaded = True
            self.active_models.append(self.densenet)
            print("[Ensemble] DenseNet loaded successfully.")
        except Exception as e:
            print(f"[Ensemble] DenseNet skipped: {e}")

        # 3. ViT
        try:
            self.vit.backbone.load_state_dict(torch.load(vit_path, map_location=device))
            self.vit_loaded = True
            self.active_models.append(self.vit)
            print("[Ensemble] Vision Transformer loaded successfully.")
        except Exception as e:
            print(f"[Ensemble] ViT skipped: {e}")

        if len(self.active_models) == 0:
            print("WARNING: No models loaded into the ensemble!")

        print(f"[Ensemble] Weighted soft-voting: EfficientNet=0.60, DenseNet=0.20, ViT=0.20")

    def forward(self, x):
        if len(self.active_models) == 0:
            return torch.zeros(x.size(0), 4).to(x.device)

        weighted_probs = None
        total_weight   = 0.0

        model_weights = {
            id(self.effnet):   0.60,
            id(self.densenet): 0.20,
            id(self.vit):      0.20,
        }

        for model in self.active_models:
            w      = model_weights.get(id(model), 0.33)
            logits = model(x)
            probs  = torch.softmax(logits, dim=-1)

            if weighted_probs is None:
                weighted_probs = w * probs
            else:
                weighted_probs = weighted_probs + w * probs
            total_weight += w

        avg_probs = weighted_probs / total_weight
        pseudo_logits = torch.log(avg_probs + 1e-9)
        return pseudo_logits

def safety_gate(logits, entropy_threshold=0.8):
    """
    Calculates predictive entropy.
    If entropy is above the threshold, escalating to a human radiologist.
    """
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    
    decisions = []
    for ent in entropy:
        if ent > entropy_threshold:
            decisions.append("ESCALATE_TO_HUMAN")
        else:
            decisions.append("AUTO_APPROVE")
            
    return decisions
