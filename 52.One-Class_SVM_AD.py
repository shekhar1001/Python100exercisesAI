from sklearn.svm import OneClassSVM
import numpy as np

X_train=np.random.randn(100,2)
X_test=np.random.randn(20,2)

model=OneClassSVM(nu=0.1,kernel='rbf')
model.fit(X_train)

preds=model.predict(X_test)
print("Predictions:",preds)

# Predictions: [ 1 -1 -1  1  1  1  1  1 -1 -1  1  1 -1 -1  1 -1  1  1  1  1]