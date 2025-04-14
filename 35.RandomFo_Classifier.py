from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y= load_iris(return_X_y=True)
X_train, X_test, y_train, y_test=train_test_split(X,y)

model= RandomForestClassifier(n_estimators=100)

model.fit(X_train,y_train)

print("Random Forest Accuracy", model.score(X_test,y_test))