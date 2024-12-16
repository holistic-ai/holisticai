import matplotlib.pyplot as plt
import numpy as np
import problexity as px
from sklearn.datasets import make_friedman1, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Regression Synthetic Datasets
def generate_friedman1():
    X, y = make_friedman1(n_samples=1000, noise=0.5, random_state=42)
    return X[:, :2], y  # Use only the first 2 features

def generate_regression1():
    return make_regression(n_samples=1000, n_features=2, noise=0.5, random_state=42)

def generate_quadratic():
    X = np.random.rand(1000, 2) * 10 - 5
    y = X[:, 0]**2 + X[:, 1]**2 + np.random.normal(0, 0.5, 1000)
    return X, y

def generate_sine_wave():
    X = np.random.rand(1000, 2) * 10 - 5
    y = np.sin(X[:, 0]) + np.sin(X[:, 1]) + np.random.normal(0, 0.5, 1000)
    return X, y

def generate_exponential():
    X = np.random.rand(1000, 2) * 10 - 5
    y = np.exp(X[:, 0]) + np.exp(X[:, 1]) + np.random.normal(0, 0.5, 1000)
    return X, y

def generate_cubic():
    X = np.random.rand(1000, 2) * 10 - 5
    y = X[:, 0]**3 + X[:, 1]**3 + np.random.normal(0, 0.5, 1000)
    return X, y

def generate_harmonic():
    X = np.random.rand(1000, 2) * 10 - 5
    y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + np.random.normal(0, 0.5, 1000)
    return X, y

def generate_multiplicative():
    X = np.random.rand(1000, 2) * 10 - 5
    y = X[:, 0] * X[:, 1] + np.random.normal(0, 0.5, 1000)
    return X, y

def generate_logarithmic():
    X = np.random.rand(1000, 2) * 10 + 0.1  # Shift to avoid log(0)
    y = np.log(X[:, 0]) + np.log(X[:, 1]) + np.random.normal(0, 0.5, 1000)
    return X, y

def generate_inverse():
    X = np.random.rand(1000, 2) * 10 + 0.1  # Shift to avoid division by zero
    y = 1 / (X[:, 0] + 0.1) + 1 / (X[:, 1] + 0.1) + np.random.normal(0, 0.5, 1000)
    return X, y


# Dataset generators dictionary
dataset_generators = [
    ("Friedman 1", generate_friedman1),
    ("Linear Regression", generate_regression1),
    ("Quadratic", generate_quadratic),
    ("Sine Wave", generate_sine_wave),
    ("Exponential", generate_exponential),
    ("Cubic", generate_cubic),
    ("Harmonic", generate_harmonic),
    ("Multiplicative", generate_multiplicative),
    ("Logarithmic", generate_logarithmic),
    ("Inverse", generate_inverse)
]


# Functions for data preprocessing and model evaluation
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)

def compute_complexity(X_train, y_train):
    cc = px.ComplexityCalculator(mode='regression')
    cc.fit(X_train, y_train)
    s3 = cc.report()['complexities']['s3']
    return cc, s3

def train_and_evaluate_model(X_train, X_test, y_train, y_test, s3):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    src = r2 * s3
    return r2, src

# Main function for dataset evaluation with results stored in a data structure
def evaluate_datasets(dataset_generators):
    results = []  # List to store results for each dataset
    fig, axs = plt.subplots(len(dataset_generators), 2, figsize=(14, 7 * len(dataset_generators)))

    for idx, (name, generator) in enumerate(dataset_generators):
        dataset_result = {"name": name, "r2": None, "s3": None, "src": None, "error": None}
        try:
            # Generate dataset
            X, y = generator()

            # Preprocess data
            X_train, X_test, y_train, y_test = preprocess_data(X, y)

            # Compute complexity and train the model
            cc, s3 = compute_complexity(X_train, y_train)
            r2, src = train_and_evaluate_model(X_train, X_test, y_train, y_test, s3)

            # Store results
            dataset_result.update({"r2": r2, "s3": s3, "src": src})

            # Plot dataset
            axs[idx, 0].scatter(X[:, 0], y, c=y, cmap='viridis')
            axs[idx, 0].set_title(f'{name} Dataset')

            # Generate complexity plot
            cc.plot(fig, (len(dataset_generators), 2, 2 * idx + 2))
            axs[idx, 1].set_title(f'Complexity Plot\nR^2: {r2:.4f}, SRC: {src:.4f}')

        except Exception as e:
            # Store error details
            dataset_result["error"] = str(e)

        # Append the result for the current dataset
        results.append(dataset_result)

    plt.tight_layout()
    plt.show()

    return [results]

