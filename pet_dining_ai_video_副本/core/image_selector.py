
import os
import random
from config import ANIMAL_IMAGE_DIR

def select_random_animal_image(image_dir: str = ANIMAL_IMAGE_DIR) -> str:
    subfolders = [f for f in os.listdir(image_dir)
                  if os.path.isdir(os.path.join(image_dir, f))]

    if not subfolders:
        raise FileNotFoundError("No animal folders found.")

    selected_folder = random.choice(subfolders)
    folder_path = os.path.join(image_dir, selected_folder)
    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not images:
        raise FileNotFoundError(f"No images in folder: {selected_folder}")

    return os.path.join(folder_path, random.choice(images))
