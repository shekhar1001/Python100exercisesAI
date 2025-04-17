from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

X,y=load_iris(return_X_y=True)

tsne=TSNE(n_components=2,perplexity=30)
X_tsne=tsne.fit_transform(X)

plt.scatter(X_tsne[:,0],X_tsne[:,1],c=y,cmap="viridis")
plt.title("t-SNE of Iris Dataset")
plt.show()