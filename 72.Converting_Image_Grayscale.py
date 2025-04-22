import cv2
import matplotlib.pyplot as plt

# Read the image (ensure 'sample.png' is in the same directory)
img = cv2.imread('sample.png')

# Check if image was successfully loaded
if img is None:
    print("Error: Image not found or cannot be opened.")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display using matplotlib
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')
    plt.show()
