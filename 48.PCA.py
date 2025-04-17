from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

X, _=load_iris(return_X_y=True)
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X)

print("Reduced Shape:",X_pca.shape)
