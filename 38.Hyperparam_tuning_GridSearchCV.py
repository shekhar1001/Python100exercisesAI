from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
y = [0, 1, 0, 1, 0, 1]

grid = GridSearchCV(SVC(), param_grid=params, cv=2)
grid.fit(X, y)

print("Best Params:", grid.best_params_)

# Best Params: {'C': 0.1, 'kernel': 'linear'}
