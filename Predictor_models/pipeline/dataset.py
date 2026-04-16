import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from .config import PATHS

class EndoDataset(Dataset):
    """
    Dataset genérico para clasificación binaria en endoscopía.
    
    target_category: 'polipos', 'sangre', 'inflamacion', 'negativos'
    transform: transformaciones de torchvision
    include_others_as_negative: Si es True, usa las otras categorías patológicas como Label 0.
    """
    def __init__(self, target_category, transform=None, include_others_as_negative=True):
        self.target_category = target_category
        self.transform = transform
        self.samples = []
        
        # 1. Cargar muestras de la categoría OBJETIVO (Etiqueta 1)
        target_path = PATHS[target_category]
        self._add_from_dir(target_path, label=1)
        
        # 2. Cargar muestras de la categoría NEGATIVA (Etiqueta 0)
        # Siempre incluimos 'negativos' (casos normales) como Label 0
        self._add_from_dir(PATHS['negativos'], label=0)
        
        # Si se desea, incluimos las otras patologías como Label 0 (One-vs-Rest)
        if include_others_as_negative:
            for cat, path in PATHS.items():
                if cat != target_category and cat != 'negativos':
                    self._add_from_dir(path, label=0)
                    
    def _add_from_dir(self, directory, label):
        path = Path(directory)
        if not path.exists():
            print(f"Advertencia: La ruta {directory} no existe.")
            return
            
        # Extensiones comunes de imagen
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        
        # Búsqueda recursiva de imágenes
        for entry in path.rglob('*'):
            if entry.is_file() and entry.suffix.lower() in extensions:
                self.samples.append((str(entry), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)
