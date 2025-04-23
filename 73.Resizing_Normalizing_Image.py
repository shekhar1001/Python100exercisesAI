import cv2
import matplotlib.pyplot as plt

img = cv2.imread('sample.png')

if img is None:
    print("Error: Image not found or unable to load.")
else:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(img_rgb, (128, 128))

    normalized = resized / 255.0
    print("Shape:", normalized.shape)
    plt.imshow(normalized)
    plt.title("Resized & Normalized Image (128x128)")
    plt.axis("off")
    plt.show()
