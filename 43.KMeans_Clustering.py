from sklearn.cluster import KMeans
import numpy as np

# Creating Dummy Data
X=np.array([[1,2],[1,4],[1,0],
            [4,2],[4,4],[4,0]])

model=KMeans(n_clusters=2,random_state=42)
model.fit(X)

print("Cluster Labels:",model.labels_)
print("Cluster Centers:",model.cluster_centers_)

