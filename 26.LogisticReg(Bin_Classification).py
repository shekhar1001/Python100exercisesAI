from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris=load_iris()
iris.head()
X=iris.data[iris.target!=2]
y=iris.target[iris.target!=2]

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.25)

model=LogisticRegression()
model.fit(X_train,y_train)

print("Accuracy",model.score(X_test,y_test))
