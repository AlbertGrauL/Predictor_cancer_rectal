import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name="efficientnet_b0", pretrained=True):
    """
    Factoría de modelos. 
    Devuelve un modelo con la cabeza modificada para clasificación binaria.
    """
    if model_name == "efficientnet_b0":
        # Usamos los pesos por defecto (ImageNet) si pretrained=True
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        
        # Modificar la cabeza (classifier)
        # EfficientNet_B0 tiene un classifier que es un Dropout + Linear
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    
    else:
        raise ValueError(f"Modelo {model_name} no soportado.")
        
    return model
