from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split

# Generating synthetic binary classification data
X_train = np.random.rand(1000, 4)
y_train = (X_train.sum(axis=1) > 2).astype(int)

X_test = np.random.rand(200, 4)
y_test = (X_test.sum(axis=1) > 2).astype(int)


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# Building a simple neural network
model = Sequential([
    Dense(10, activation='relu', input_shape=(4,)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])

# Output
# .
# .
#  1/20 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - accuracy: 1.0000 - loss:20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9520 - loss: 0.3541 - val_accuracy: 0.9375 - val_loss: 0.3487
# Epoch 50/50
#  1/20 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - accuracy: 1.0000 - loss:20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9348 - loss: 