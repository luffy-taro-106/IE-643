import os
from PIL import Image
import torchvision.transforms as transforms

def load_image(image_path):
    """Load an image from the specified path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = Image.open(image_path).convert("RGB")
    return image

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess the image for the model."""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(image)

def save_image(image_tensor, save_path):
    """Save a tensor as an image."""
    image = image_tensor.clone().detach().cpu()
    image = transforms.ToPILImage()(image)
    image.save(save_path)