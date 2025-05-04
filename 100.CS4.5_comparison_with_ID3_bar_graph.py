import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import math

# Load and preprocess Titanic dataset
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Fare"]]
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Age"] = pd.cut(df["Age"], bins=3, labels=[0, 1, 2])
    df["Fare"] = pd.cut(df["Fare"], bins=3, labels=[0, 1, 2])
    return df

# Entropy
def entropy(y):
    probs = y.value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in probs if p > 0)

# ID3 - Information Gain
def info_gain(data, attr, target):
    total_entropy = entropy(data[target])
    vals = data[attr].unique()
    weighted_entropy = sum(
        (len(subset) / len(data)) * entropy(subset[target])
        for val in vals if not (subset := data[data[attr] == val]).empty
    )
    return total_entropy - weighted_entropy

# C4.5 - Gain Ratio
def gain_ratio(data, attr, target):
    gain = info_gain(data, attr, target)
    split_info = entropy(data[attr])
    return gain / split_info if split_info else 0

# Build decision tree (generic)
def build_tree(data, features, target, method="id3"):
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]
    if not features:
        return data[target].mode()[0]

    if method == "id3":
        best = max(features, key=lambda f: info_gain(data, f, target))
    else:
        best = max(features, key=lambda f: gain_ratio(data, f, target))

    tree = {best: {}}
    for val in data[best].unique():
        subset = data[data[best] == val]
        if subset.empty:
            tree[best][val] = data[target].mode()[0]
        else:
            tree[best][val] = build_tree(
                subset.drop(columns=[best]), [f for f in features if f != best], target, method
            )
    return tree

# Predict using tree
def predict(tree, sample):
    while isinstance(tree, dict):
        attr = next(iter(tree))
        val = sample.get(attr)
        if val not in tree[attr]:
            return 0  # default
        tree = tree[attr][val]
    return tree

# Evaluate model
def evaluate(tree, test_data):
    y_true = test_data["Survived"]
    y_pred = [predict(tree, row) for _, row in test_data.iterrows()]
    return accuracy_score(y_true, y_pred), y_pred

# Main
def main():
    df = load_data()
    features = [col for col in df.columns if col != "Survived"]
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    print("Training ID3...")
    id3_tree = build_tree(train.copy(), features.copy(), "Survived", method="id3")
    id3_acc, _ = evaluate(id3_tree, test)

    print("Training C4.5...")
    c45_tree = build_tree(train.copy(), features.copy(), "Survived", method="c45")
    c45_acc, _ = evaluate(c45_tree, test)

    # Bar chart comparison
    plt.bar(["ID3", "C4.5"], [id3_acc, c45_acc], color=["blue", "green"])
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    for i, acc in enumerate([id3_acc, c45_acc]):
        plt.text(i, acc + 0.01, f"{acc:.2f}", ha='center')
    plt.show()

if __name__ == "__main__":
    main()
