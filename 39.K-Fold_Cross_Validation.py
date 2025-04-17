from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X,y=load_iris(return_X_y=True)
model=LogisticRegression(max_iter=200)

scores=cross_val_score(model,X,y,cv=5)
print("Cross-validation scores:",scores)
print("Average accuracy",scores.mean())

# Cross-validation scores: [0.96666667 1.         0.93333333 0.96666667 1.        ]
# Average accuracy 0.9733333333333334