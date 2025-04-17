from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

X=np.random.rand(10,2)

linked=linkage(X,'ward')
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()
