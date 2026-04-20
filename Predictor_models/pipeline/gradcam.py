import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class GradCAM:
    """Implementación básica de Grad-CAM para detectar qué zonas activan la predicción."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        
        # Asumiendo clasificación binaria (logit único)
        self.model.zero_grad()
        output.backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        grad_cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        grad_cam = F.relu(grad_cam)
        grad_cam = grad_cam.detach().cpu().numpy()
        grad_cam = cv2.resize(grad_cam, (224, 224))
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)
        
        return grad_cam

def visualize_gradcam(image_path, model, target_layer, output_path):
    # Cargar y preprocesar imagen
    from .transforms import get_transforms
    transform = get_transforms(train=False)
    
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)
    
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate_heatmap(input_tensor)
    
    # Overlay
    img_array = np.array(img.resize((224, 224)))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
    
    plt.imsave(output_path, overlay)
    print(f"Visualización guardada en {output_path}")
