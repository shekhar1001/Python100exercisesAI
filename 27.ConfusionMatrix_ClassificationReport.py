from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


X,y=load_iris(return_X_y=True)
X_train, X_test, y_train, y_test= train_test_split(X,y)

model=LogisticRegression(max_iter=200)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("\nClassification Report:\n", classification_report(y_test,y_pred))
