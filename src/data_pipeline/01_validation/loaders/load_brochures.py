from PIL import Image
import numpy as np


def load_brochure(image_path):

    image = Image.open(image_path)

    image_np = np.array(image)

    print("\n===== BROCHURE INFO =====")
    print(f"Shape : {image_np.shape}")

    return image_np