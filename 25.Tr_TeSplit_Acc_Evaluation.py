from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import numpy as np

X=np.array([[1],[2],[3],[4],[5]])
y=np.array([4,8,12,16,20])

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.25)

model=LinearRegression()
model.fit(X_train,y_train)

predictions=model.predict(X_test)
print("MSE:",mean_squared_error(y_test,predictions))

