from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

X,y=make_classification(n_samples=1000,n_classes=2,n_informative=2)
X_train,X_test,y_train,y_test=train_test_split(X,y)

model=RandomForestClassifier()
model.fit(X_train,y_train)

probs=model.predict_proba(X_test)[:, 1]

fpr, tpr, _=roc_curve(y_test,probs)
auc_score=roc_auc_score(y_test,probs)

plt.plot(fpr, tpr, label=f"AUC={auc_score:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve")
plt.show()