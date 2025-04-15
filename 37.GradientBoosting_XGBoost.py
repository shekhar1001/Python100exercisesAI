from xgboost import XGBClassifier
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split

X,y= load_iris(return_X_y=True)
X_train, X_test, y_train, y_test=train_test_split

model=XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train,y_train)

print("XGBoost Accuracy", model.score(X_test,y_test))

