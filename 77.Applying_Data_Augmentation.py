import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Data augmentation for training
aug_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Keep validation split consistent
)

# Augmented training data generator
train_gen = aug_gen.flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'   # Use subset for training data
)

# Validation data generator (no augmentation, just rescale)
val_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Use subset for validation data
)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train model with augmented data
model.fit(train_gen, validation_data=val_gen, epochs=10)
