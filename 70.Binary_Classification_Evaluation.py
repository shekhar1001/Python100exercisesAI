import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(16, input_shape=(X.shape[1],), activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC Score: {roc_auc:.4f}")

# Output:
# .
# .
# 46/46 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9990 - loss: 0.0044
# 1/4 ━━━━━━━━━━━━━━━━━━━━ 0s 199ms/step - accuracy: 0.9688 - loss4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - accuracy: 0.9738 - loss: 0.1262 

# Test Accuracy: 0.9737
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step

# Confusion Matrix:
# [[41  2]
#  [ 1 70]]

# Classification Report:
#               precision    recall  f1-score   support

#            0       0.98      0.95      0.96        43
#            1       0.97      0.99      0.98        71

#     accuracy                           0.97       114
#    macro avg       0.97      0.97      0.97       114
# weighted avg       0.97      0.97      0.97       114

# ROC AUC Score: 0.9944