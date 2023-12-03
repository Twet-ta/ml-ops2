# infer.py
from pickle import load

import dvc.api
import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split


def save_predictions(X_test, y_test, predictions):
    # Save predictions to results.csv
    with open("./results.csv", "w") as file:
        file.write("x, y_true, y_pred\n")
        for X_i, y_i, y_pred_i in zip(X_test.values, y_test, predictions):
            s = str(X_i) + "," + str(y_i) + "," + str(y_pred_i) + "\n"
            file.write(s)


def print_metrics(y_test, predictions):
    # Print evaluation metrics
    print("accuracy: ", accuracy_score(y_test, predictions))
    print("f1: ", f1_score(y_test, predictions, average="micro"))
    print("precision: ", precision_score(y_test, predictions, average="micro"))


@hydra.main(config_path="configs", config_name="post", version_base="2.1")
def infer(cfg: DictConfig):
    # Load data
    with dvc.api.open("data/iris_dataset.csv") as fd:
        df = pd.read_csv(fd)

    X = df.drop("target", axis=1)
    y = df["target"]

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    # Load model
    with open("model/iris_model.pkl", "rb") as file:
        trained_model = load(file)

    # Predict from the test dataset
    predictions = trained_model.predict(X_test)

    if cfg.postprocess.save_predictions:
        # Save predictions if configured
        save_predictions(X_test, y_test, predictions)

    if cfg.postprocess.visualize_results:
        # Visualize results if configured
        visualize_results(y_test, predictions)

    # Print metrics
    print_metrics(y_test, predictions)


def visualize_results(y_true, y_pred):
    # Add visualization code here based on your needs
    pass


if __name__ == "__main__":
    infer()
