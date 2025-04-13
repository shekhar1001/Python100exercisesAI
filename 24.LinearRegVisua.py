import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

X=np.array([[1],[2],[3],[4]])
y=np.array([2,4,6,8])

model=LinearRegression()
model.fit(X,y)

plt.scatter(X,y)
plt.plot(X,model.predict(X),color='red')
plt.title("Linear Regression Visualization")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
