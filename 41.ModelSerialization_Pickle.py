import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

X=np.array([[1],[2],[3]])
y=np.array([[2],[4],[6]])

model=LinearRegression()
model.fit(X,y)

# Saving the model
with open("linear_model.pkl","wb") as f:
    pickle.dump(model,f)

# Loading the model
with open("Linear_model.pkl","rb") as f:
    loaded_model=pickle.load(f)

print("Predictions:",loaded_model.predict([[4]]))

