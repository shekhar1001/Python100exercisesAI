from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential([
    Dense(10, activation='tanh',input_shape=(3,)),
    Dense(1, activation='sigmoid')
])

model.summary()

