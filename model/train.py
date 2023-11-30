from pickle import dump

import dvc.api
import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    # Load data
    with dvc.api.open("data/iris_dataset.csv") as fd:
        df = pd.read_csv(fd)

    # Preprocess data
    preprocess_cfg = cfg.preprocess
    columns_to_use = preprocess_cfg.columns
    X = df[columns_to_use]
    y = df["target"]

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    # Train model
    svc = SVC(C=cfg.train.C, kernel=cfg.train.kernel)
    svc.fit(X_train, y_train)

    # Save model to DVC
    with open("model/iris_model.pkl", "wb") as fd:
        dump(svc, fd)


if __name__ == "__main__":
    train()
