import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X,y =load_iris(return_X_y=True)
model=RandomForestClassifier()
model.fit(X,y)

features=pd.Series(model.feature_importances_,index=load_iris().feature_names)
features.sort_values().plot(kind='barh', color='skyblue')
plt.title("Feature Importance")
plt.show()