from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


import numpy as np
X=np.random.rand(500,5)
y=np.random.randint(0,3,500)
y_cat=to_categorical(y)

model=Sequential([
    Dense(10, activation='relu',input_shape=(5,)),
    Dense(3, activation='softmax')

])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y_cat, epochs=10)

# Output
# .
# .
# Epoch 9/10
#  1/16 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step - accuracy: 0.2188 - loss:16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.3242 - loss: 1.1106 
# Epoch 10/10
#  1/16 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - accuracy: 0.2812 - loss:16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.3629 - loss: 1.1053 