import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

X=np.array([[1,2],[1,4],[1,0],
            [4,2],[4,4],[4,0]])

model=KMeans(n_clusters=2,random_state=42)
model.fit(X)

plt.scatter(X[:,0],X[:,1],c=model.labels_,cmap='viridis')
plt.scatter(*model.cluster_centers_.T,color="red",marker='x',s=200)
plt.title("KMeans Clustering")
plt.show()