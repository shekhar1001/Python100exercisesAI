import pandas as pd
import numpy as np
from collections import Counter
from math import log2
from graphviz import Digraph

# Load Titanic dataset
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
    df = df.dropna()
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Age'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4])
    df['Fare'] = pd.qcut(df['Fare'], 4, labels=[0, 1, 2, 3])
    return df

# Entropy
def entropy(y):
    counts = Counter(y)
    total = len(y)
    return -sum((count/total) * log2(count/total) for count in counts.values())

# Information Gain
def info_gain(data, attr, target='Survived'):
    total_entropy = entropy(data[target])
    vals = data[attr].unique()
    weighted = sum((len(sub := data[data[attr] == v]) / len(data)) * entropy(sub[target]) for v in vals)
    return total_entropy - weighted

# ID3 Recursive Tree Builder
def id3(data, features, target='Survived'):
    labels = data[target]
    if len(set(labels)) == 1:
        return labels.iloc[0]
    if not features:
        return Counter(labels).most_common(1)[0][0]

    gains = [info_gain(data, f, target) for f in features]
    best = features[np.argmax(gains)]
    tree = {best: {}}
    
    for val in sorted(data[best].unique()):
        sub = data[data[best] == val]
        subtree = id3(sub, [f for f in features if f != best], target)
        tree[best][val] = subtree
    return tree

# Prediction
def predict(tree, row):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    val = row[attr]
    subtree = tree[attr].get(val)
    return predict(subtree, row) if subtree else 0

# Visualizing ID3 Tree using graphviz
def visualize_tree(tree, dot=None, parent=None, edge_label=""):
    if dot is None:
        dot = Digraph()
        dot.attr("node", shape="box", style="filled", color="lightblue2")

    if not isinstance(tree, dict):
        node_id = str(id(tree))
        dot.node(node_id, label=str(tree), color="lightgreen")
        if parent:
            dot.edge(parent, node_id, label=edge_label)
        return dot

    attr = next(iter(tree))
    node_id = str(id(tree))
    dot.node(node_id, label=attr)
    if parent:
        dot.edge(parent, node_id, label=edge_label)

    for val, subtree in tree[attr].items():
        visualize_tree(subtree, dot, node_id, edge_label=str(val))
    return dot

# Main
def main():
    df = load_data()
    features = [col for col in df.columns if col != 'Survived']
    tree = id3(df, features)
    
    print("Training on ID3 complete.")
    
    predictions = df.apply(lambda row: predict(tree, row), axis=1)
    accuracy = np.mean(predictions == df['Survived'])
    print(f"Training Accuracy: {accuracy:.4f}")
    print("\nSample Predictions:\n", pd.DataFrame({
        "Actual": df['Survived'].values[:10],
        "Predicted": predictions.values[:10]
    }))

    # Visualizing and render the tree
    dot = visualize_tree(tree)
    dot.render("id3_titanic_tree", format="png", cleanup=True)
    dot.view()

if __name__ == "__main__":
    main()
