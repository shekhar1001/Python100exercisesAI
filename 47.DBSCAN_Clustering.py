from sklearn.cluster import DBSCAN
import numpy as np

X=np.random.rand(20,2)
model=DBSCAN(eps=0.2,min_samples=2)
labels=model.fit_predict(X)
print("Cluster labels",labels)

# Cluster labels [ 0  0  1 -1  0  1  2  1  2  0  0  1  0  0  0  1  1  0  0  0]