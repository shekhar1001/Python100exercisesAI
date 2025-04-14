import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

X,y= make_classification(n_features=2, n_redundant=0,n_informative=2,n_clusters_per_class=1)
model=SVC(kernel='linear')
model.fit(X,y)

import numpy as np
w=model.coef_[0]
b=model.intercept_[0]
x_points=np.linspace(min(X[:,0]),max(X[:,0]))
y_points=-(w[0]/w[1])*x_points-b/w[1]

plt.scatter(X[:, 0],X[:, 1],c=y)
plt.plot(x_points,y_points,color='red')
plt.title("SVM Decision Boundry")
plt.show()




