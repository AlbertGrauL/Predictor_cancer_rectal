from __future__ import annotations


def disable_inplace_activations(module) -> None:
    for child in module.modules():
        if hasattr(child, "inplace"):
            try:
                child.inplace = False
            except Exception:
                pass


def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    try:
        import torch.nn as nn
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError("torch/torchvision no estan instalados. Ejecuta `uv sync`.") from exc

    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(
            f"Modelo no soportado: {model_name}. "
            "Modelos activos: resnet50, efficientnet_b0, densenet121."
        )
    disable_inplace_activations(model)
    return model


def freeze_backbone(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if hasattr(model, "fc"):
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    elif hasattr(model, "classifier"):
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True


def unfreeze_all(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
