# AI Project 2 - Data Classification

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Dataset

iris = load_iris()
print(iris.feature_names)
print(iris.target_names)

X = iris.data

y = iris.target

# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create AI Model


model = KNeighborsClassifier(n_neighbors=3)

# Train Model

model.fit(X_train, y_train)

# Make Predictions

predictions = model.predict(X_test)

# Calculate Accuracy

accuracy = accuracy_score(y_test, predictions)

# Print Results

print("Predictions:", predictions)
print("Actual Values:", y_test)
print("Model Accuracy:", accuracy * 100, "%")