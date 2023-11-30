from pickle import load

import dvc.api
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split


def load_data():
    # Load the dataset locally or through DVC
    with dvc.api.open_url("model/iris_dataset.csv", repo="myremote") as fd:
        df = pd.read_csv(fd)

    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def load_model():
    # Load the trained Support Vector Machine model using DVC
    with dvc.api.open_url("model/iris_model.pkl", repo="myremote") as file:
        model = load(file)
    return model


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


if __name__ == "__main__":
    # Load data
    X, y = load_data()

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    # Load model
    trained_model = load_model()

    # Predict from the test dataset
    predictions = trained_model.predict(X_test)

    # Save predictions
    save_predictions(X_test, y_test, predictions)

    # Print metrics
    print_metrics(y_test, predictions)
