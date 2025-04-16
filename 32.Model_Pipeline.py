from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X,y=load_iris(return_X_y=True)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

pipe=Pipeline([
('Scaler', StandardScaler()),
('clf', LogisticRegression(max_iter=200))
])

pipe.fit(X_train,y_train)
print("Pipeline Accuracy", pipe.score(X_test,y_test))