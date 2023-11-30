from pickle import dump

import dvc.api
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def load_data():
    data_dvc_path = "data/iris_dataset.csv"
    remote_name = "myremote"
    with dvc.api.open(data_dvc_path, repo=remote_name) as fd:
        df = pd.read_csv(fd)

    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def train_model(X_train, y_train):
    svc = SVC()
    svc.fit(X_train, y_train)
    return svc


def save_model(model):
    # Save the trained model using DVC
    with open("model/iris_model.pkl", "wb") as fd:
        dump(model, fd)


if __name__ == "__main__":
    # Load data
    X, y = load_data()

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    # Train model
    trained_model = train_model(X_train, y_train)

    # Save model
    save_model(trained_model)
