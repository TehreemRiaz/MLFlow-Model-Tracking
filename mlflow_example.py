import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Set up MLflow experiment
experiment_name = "Iris Logistic Regression"
mlflow.set_experiment(experiment_name)

param_grid = {
    "C": [0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"],
    "max_iter": [100, 200]
}

best_accuracy = 0
best_model = None
best_run_id = None

# Step 3: Hyperparameter tuning
parameter_combinations = list(ParameterGrid(param_grid))

for params in parameter_combinations:
    with mlflow.start_run():
        # Train model with current set of parameters
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters, metrics, and the model to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        # Log the model with input example (to resolve signature warning)
        input_example = np.array(X_test[0]).reshape(1, -1)  # Example input (reshape for a single prediction)
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)
        
        # Track the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_run_id = mlflow.active_run().info.run_id

        # Log confusion matrix as an artifact
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

# Step 4: Register the best model in the MLflow Model Registry
model_uri = f"runs:/{best_run_id}/model"
print("Best Model URI:", model_uri)
mlflow.register_model(model_uri, "Iris_Classification_Model")
