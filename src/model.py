import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 7

def build_model(trainable_base: bool = False) -> nn.Module:
    """
    Construye el modelo usando EfficientNetB0 preentrenado en ImageNet.
    - trainable_base=False: solo entrena la cabeza (fase 1)
    - trainable_base=True: entrena todo el modelo (fase 2)
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Congelar o descongelar el backbone
    for param in model.parameters():
        param.requires_grad = trainable_base

    # Reemplazar la cabeza de clasificacion
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES)
    )

    return model


def get_device() -> torch.device:
    """Detecta si hay GPU disponible, sino usa CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando: {device}")
    return device