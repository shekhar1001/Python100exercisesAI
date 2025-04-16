from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

X=np.random.rand(100,2)

inertia=[]

for k in range(1,10):
    kmeans=KMeans(n_clusters=k,random_state=42).fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1,10),inertia,marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()