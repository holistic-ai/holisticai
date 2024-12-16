import matplotlib.pyplot as plt
import numpy as np
import problexity as px
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_classification,
    make_friedman1,
    make_gaussian_quantiles,
    make_moons,
    make_swiss_roll,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Dataset generation functions
def generate_moons():
    return make_moons(n_samples=1000, noise=0.2, random_state=42)

def generate_circles():
    return make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=42)

def generate_classification():
    return make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

def generate_blobs():
    return make_blobs(n_samples=1000, centers=3, cluster_std=2.5, n_features=2, random_state=42)

def generate_xor():
    X = np.random.randn(200, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    return X, y

def generate_swiss_roll():
    X, _ = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
    y = (X[:, 0] > 0).astype(int)
    return X[:, [0, 2]], y

def generate_gaussian_quantiles():
    return make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=2, random_state=42)

def generate_friedman1():
    X, y = make_friedman1(n_samples=1000, noise=0.2, random_state=42)
    y = (y > np.median(y)).astype(int)  # Convert continuous target to binary
    return X[:, :2], y

def generate_spirals():
    X1 = np.array([[i / 500 * np.cos(1.75 * i / 500 * 2 * np.pi),
                    i / 500 * np.sin(1.75 * i / 500 * 2 * np.pi)] for i in range(500)])
    X2 = np.array([[-i / 500 * np.cos(1.75 * i / 500 * 2 * np.pi),
                    -i / 500 * np.sin(1.75 * i / 500 * 2 * np.pi)] for i in range(500)])
    X = np.vstack((X1, X2))
    y = np.array([0] * 500 + [1] * 500)
    return X, y

def generate_two_intertwined_spirals():
    n = 500
    theta = np.linspace(0, 4 * np.pi, n)
    r = np.linspace(0, 1, n)
    x1 = r * np.sin(theta) + np.random.normal(scale=0.1, size=n)
    y1 = r * np.cos(theta) + np.random.normal(scale=0.1, size=n)
    x2 = -r * np.sin(theta) + np.random.normal(scale=0.1, size=n)
    y2 = -r * np.cos(theta) + np.random.normal(scale=0.1, size=n)
    X = np.vstack((np.c_[x1, y1], np.c_[x2, y2]))
    y = np.array([0] * n + [1] * n)
    return X, y


# Dataset generators list
dataset_generators = [
    ("Make Moons", generate_moons),
    ("Make Circles", generate_circles),
    ("Make Classification", generate_classification),
    ("Make Blobs", generate_blobs),
    ("XOR", generate_xor),
    ("Swiss Roll", generate_swiss_roll),
    ("Gaussian Quantiles", generate_gaussian_quantiles),
    ("Make Friedman 1", generate_friedman1),
    ("Spirals", generate_spirals),
    ("Two Intertwined Spirals", generate_two_intertwined_spirals)
]


# Preprocessing and evaluation functions
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)

def compute_complexity(X_train, y_train):
    cc = px.ComplexityCalculator()
    cc.fit(X_train, y_train)
    t1 = cc.report()['complexities']['t1']
    return cc, t1

def train_and_evaluate_model(X_train, X_test, y_train, y_test, t1):
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    src = acc / t1
    return acc, src

# Main evaluation function
def evaluate_datasets(dataset_generators):
    results = []  # To store results for each dataset
    fig, axs = plt.subplots(len(dataset_generators), 2, figsize=(14, 7 * len(dataset_generators)))

    for idx, (name, generator) in enumerate(dataset_generators):
        dataset_result = {"name": name, "accuracy": None, "t1": None, "src": None, "error": None}
        try:
            # Generate dataset
            X, y = generator()

            # Preprocess data
            X_train, X_test, y_train, y_test = preprocess_data(X, y)

            # Compute complexity and train the model
            cc, t1 = compute_complexity(X_train, y_train)
            acc, src = train_and_evaluate_model(X_train, X_test, y_train, y_test, t1)

            # Store results
            dataset_result.update({"accuracy": acc, "t1": t1, "src": src})

            # Plot dataset
            axs[idx, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
            axs[idx, 0].set_title(f'{name} Dataset')

            # Generate complexity plot
            cc.plot(fig, (len(dataset_generators), 2, 2 * idx + 2))
            axs[idx, 1].set_title(f'Complexity Plot\nAccuracy: {acc:.4f}, MAGOC: {src:.4f}')

        except Exception as e:
            # Store error details
            dataset_result["error"] = str(e)

        # Append the result for the current dataset
        results.append(dataset_result)

    plt.tight_layout()
    plt.show()

    return results
