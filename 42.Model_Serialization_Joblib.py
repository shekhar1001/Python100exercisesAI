from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from joblib import dump, load

X,y=load_iris(return_X_y=True)
model=RandomForestClassifier()
model.fit(X,y)

dump(model,"rf_model.joblib")

loaded=load("rf_model.joblib")
print("Loaded model accuracy:",loaded.score(X,y))

# Loaded model accuracy:1.0
