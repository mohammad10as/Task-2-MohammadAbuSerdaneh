# Iris Species Classification using KNN

A foundational machine learning project that implements a K-Nearest Neighbors (KNN) classifier to predict the species of an iris flower based on its physical features. 

## 📊 Dataset Overview
The project utilizes the classic **Iris Dataset** from `sklearn.datasets`, which contains 150 samples across 3 distinct species:
* Setosa
* Versicolor
* Virginica

Each sample includes 4 features: sepal length, sepal width, petal length, and petal width.

## 🛠️ How It Works
1. **Data Splitting:** Divides the dataset into an 80% training set and a 20% testing set to ensure proper model validation.
2. **Model Training:** Initializes a `KNeighborsClassifier` configured with $n\_neighbors = 3$.
3. **Evaluation:** Uses `accuracy_score` to evaluate how accurately the model predicts unseen test data.

## 🚀 Getting Started

### Prerequisites
Make sure you have `scikit-learn` installed:
```bash
pip install scikit-learn
