import cv2
import matplotlib.pyplot as plt

img = cv2.imread('sample.png')

if img is None:
    print("Error: Image not found or cannot be opened.")
else:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')
    plt.show()
