import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setup data augmentation generator
aug_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load augmented data (no validation split here)
aug_data = aug_gen.flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Get one batch of augmented images and labels
batch_x, batch_y = next(aug_data)

# Visualize first 3 images with their labels
for i in range(3):
    plt.imshow(batch_x[i])
    plt.title(f"Label: {int(batch_y[i])}")
    plt.axis('off')
    plt.show()
