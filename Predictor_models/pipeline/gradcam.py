from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GradCamResult:
    heatmap: "object"
    probabilities: list[float]
    predicted_index: int


def find_target_layer(model):
    candidate_names = ["layer4", "features"]
    for name in candidate_names:
        if hasattr(model, name):
            target = getattr(model, name)
            if hasattr(target, "__getitem__"):
                try:
                    return target[-1]
                except Exception:
                    return target
            return target
    raise ValueError("No se pudo inferir la capa objetivo para Grad-CAM.")


def generate_gradcam(model, input_tensor):
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("torch/numpy no estan instalados. Ejecuta `uv sync`.") from exc

    target_layer = find_target_layer(model)
    activations = []
    gradients = []

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = int(probabilities.argmax(dim=1).item())
        score = logits[:, predicted_index]
        score.backward()

        activation = activations[-1]
        gradient = gradients[-1]
        pooled_gradients = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (activation * pooled_gradients).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return GradCamResult(
            heatmap=cam.astype(np.float32),
            probabilities=probabilities.squeeze(0).detach().cpu().tolist(),
            predicted_index=predicted_index,
        )
    finally:
        forward_handle.remove()
        backward_handle.remove()
