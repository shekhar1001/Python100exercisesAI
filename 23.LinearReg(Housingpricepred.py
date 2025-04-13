from sklearn.linear_model import LinearRegression
import numpy as np

X=np.array([[1000],[1500],[2000],[2500]])
y=np.array([200000,300000,400000,500000])

model=LinearRegression()
model.fit(X,y)

prediction=model.predict([[1800]])
print("Predicted Price:",prediction[0])