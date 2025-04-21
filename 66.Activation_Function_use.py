from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential([
    Dense(10, activation='tanh',input_shape=(3,)),
    Dense(1, activation='sigmoid')
])

model.summary()


# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
# ┃ Layer (type)             ┃ Output Shape       ┃    Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
# │ dense (Dense)            │ (None, 10)         │         40 │
# ├──────────────────────────┼────────────────────┼────────────┤
# │ dense_1 (Dense)          │ (None, 1)          │         11 │
# └──────────────────────────┴────────────────────┴────────────┘
#  Total params: 51 (204.00 B)
#  Trainable params: 51 (204.00 B)
#  Non-trainable params: 0 (0.00 B)