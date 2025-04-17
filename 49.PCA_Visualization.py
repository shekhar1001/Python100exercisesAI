import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

X,y =load_iris(return_X_y=True)

pca=PCA(n_components=2)
X_pca=pca.fit_transform(X)

plt.scatter(X_pca[:,0],X_pca[:,1],c=y,cmap="viridis")
plt.title("PCA of Iris Dataset")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
