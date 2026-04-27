from torchvision import transforms
from torchvision.transforms import v2
from .config import IMG_SIZE

def get_transforms(train=True):
    """Devuelve las transformaciones para entrenamiento o validación/test."""
    if train:
        return v2.Compose([
            v2.Resize(IMG_SIZE),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return v2.Compose([
            v2.Resize(IMG_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


