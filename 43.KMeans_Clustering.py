from sklearn.cluster import KMeans
import numpy as np

# Creating Dummy Data
X=np.array([[1,2],[1,4],[1,0],
            [4,2],[4,4],[4,0]])

model=KMeans(n_clusters=2,random_state=42)
model.fit(X)

print("Cluster Labels:",model.labels_)
print("Cluster Centers:",model.cluster_centers_)

# Cluster Labels: [0 1 0 1 1 0]
# Cluster Centers: [[2.         0.66666667]
#  [3.         3.33333333]]

