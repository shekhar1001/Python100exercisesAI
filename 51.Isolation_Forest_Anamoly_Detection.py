from sklearn.ensemble import IsolationForest
import numpy as np

X=np.random.randn(100,2)
X_outliers=np.random.uniform(low=-6,high=6,size=(10,2))
X_combined=np.vstack([X,X_outliers])

model=IsolationForest(contamination=0.1)
labels=model.fit_predict(X_combined)

print("Labels(1=normal, -1=anamoly):",labels)

# Labels(1=normal, -1=anamoly): [ 1  1  1  1 -1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
#   1  1  1  1  1  1 -1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
#   1  1  1 -1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
#   1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
#   1  1  1  1  1 -1 -1 -1 -1 -1 -1  1 -1 -1]