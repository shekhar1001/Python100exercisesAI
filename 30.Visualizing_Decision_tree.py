from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the dataset once
iris = load_iris()
X, y = iris.data, iris.target

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)


plt.figure(figsize=(10, 6))
plot_tree(model, filled=True, 
          feature_names=iris.feature_names, 
          class_names=iris.target_names)
plt.show()

print(iris)