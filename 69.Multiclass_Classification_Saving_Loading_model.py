import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Loading and preprocessing data
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Using sparse=False for older scikit-learn versions
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Defining model
model = Sequential([
    Dense(10, input_shape=(4,), activation='relu'),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=1)

# Evaluating model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Saving model
model.save("mcc_model.keras")

# Loading and evaluating saved model
loaded_model = load_model("mcc_model.keras")
loaded_loss, loaded_accuracy = loaded_model.evaluate(X_test, y_test)
print(f"\nLoaded Model Accuracy: {loaded_accuracy:.4f}")
