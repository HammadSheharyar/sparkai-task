import torch
import torch.nn as nn
from torchvision import models

SUPPORTED = ("efficientnet_b0", "mobilenet_v3_large", "resnet18")
HEAD_PREFIXES = ("classifier", "fc")


def build_model(name: str, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    w = "DEFAULT" if pretrained else None
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, 1))
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=w)
        m.classifier[2] = nn.Dropout(dropout)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
    elif name == "resnet18":
        m = models.resnet18(weights=w)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 1))
    else:
        raise ValueError(f"unsupported model {name}; choose from {SUPPORTED}")
    return m


def split_params(model: nn.Module):
    head, backbone = [], []
    for n, p in model.named_parameters():
        (head if n.startswith(HEAD_PREFIXES) else backbone).append(p)
    return backbone, head


def set_backbone_trainable(model: nn.Module, trainable: bool):
    for p in split_params(model)[0]:
        p.requires_grad = trainable


def gradcam_layer(model: nn.Module, name: str):
    return model.layer4[-1] if name == "resnet18" else model.features[-1]


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["model_name"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
