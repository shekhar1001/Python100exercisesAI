import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris=load_iris()
X, y=iris.data, (iris.target==0).astype(int)

model= LogisticRegression()
model.fit(X,y)

joblib.dump(model, 'iris_model.pkl')
