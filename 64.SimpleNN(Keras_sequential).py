from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Simple feedforward network with 1 hidden layer

model=Sequential([
    Dense(10, activation='relu', input_shape=(4,)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_corssentropy', metrics=['accuracy'])
model.summary()

# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
# ┃ Layer (type)             ┃ Output Shape       ┃    Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
# │ dense (Dense)            │ (None, 10)         │         50 │
# ├──────────────────────────┼────────────────────┼────────────┤
# │ dense_1 (Dense)          │ (None, 1)          │         11 │
# └──────────────────────────┴────────────────────┴────────────┘
#  Total params: 61 (244.00 B)
#  Trainable params: 61 (244.00 B)
#  Non-trainable params: 0 (0.00 B)