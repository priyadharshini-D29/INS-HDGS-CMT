import cv2
import matplotlib.pyplot as plt


def show_brochure(image_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    plt.title(image_path.name)
    plt.axis("off")
    plt.show()
