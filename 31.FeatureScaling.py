# Standardization

from sklearn.preprocessing import StandardScaler
import numpy as np

data=np.array([[1,200],[2,300],[3,400]])

scaler=StandardScaler()
scaled_data=scaler.fit_transform(data)

print("Standardized Data:\n", scaled_data)

