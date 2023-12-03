import subprocess
from datetime import datetime
from pathlib import Path
from pickle import dump

import dvc.api
import hydra
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def custom_precision_recall(y_true, y_probabilities):
    num_classes = y_probabilities.shape[1]
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []

    for class_label in range(num_classes):
        class_probs = y_probabilities[:, class_label]

        for threshold in thresholds:
            y_pred = (class_probs > threshold).astype(int)
            true_positive = sum((y_true == class_label) & (y_pred == 1))
            false_positive = sum((y_true != class_label) & (y_pred == 1))
            false_negative = sum((y_true == class_label) & (y_pred == 0))

            precision = (
                true_positive / (true_positive + false_positive)
                if (true_positive + false_positive) > 0
                else 0
            )
            recall = (
                true_positive / (true_positive + false_negative)
                if (true_positive + false_negative) > 0
                else 0
            )

            precisions.append(precision)
            recalls.append(recall)

    return precisions, recalls


def custom_roc_curve(y_true, y_probabilities):
    num_classes = y_probabilities.shape[1]
    thresholds = np.linspace(0, 1, 100)
    all_fpr = []
    all_tpr = []

    for class_label in range(num_classes):
        class_probs = y_probabilities[:, class_label]
        fpr_list = []
        tpr_list = []

        for threshold in thresholds:
            y_pred = (class_probs > threshold).astype(int)
            true_positive = np.sum((y_true == 1) & (y_pred == 1))
            false_positive = np.sum((y_true == 0) & (y_pred == 1))
            true_negative = np.sum((y_true == 0) & (y_pred == 0))
            false_negative = np.sum((y_true == 1) & (y_pred == 0))

            sensitivity = true_positive / (true_positive + false_negative)
            specificity = true_negative / (true_negative + false_positive)
            fpr_list.append(1 - specificity)
            tpr_list.append(sensitivity)
        all_fpr.append(fpr_list)
        all_tpr.append(tpr_list)

    return all_fpr, all_tpr


def custom_macro_average(precisions, recalls):
    # Кастомная функция для макро-усреднения Precision и Recall
    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)

    return macro_precision, macro_recall


def custom_macro_average_roc_auc(y_true, y_probabilities):
    num_classes = y_probabilities.shape[1]
    thresholds = np.linspace(0, 1, 100)
    all_auc = []

    for class_label in range(num_classes):
        class_probs = y_probabilities[:, class_label]
        auc_list = []

        for threshold in thresholds:
            y_pred = (class_probs > threshold).astype(int)
            auc = roc_auc_score(y_true, y_pred)
            auc_list.append(auc)

        all_auc.append(auc_list)

    macro_auc = np.mean(all_auc, axis=0)

    return macro_auc


@hydra.main(config_path="configs", config_name="config", version_base="2.1")
def train(cfg: DictConfig):
    # Load data
    mlflow.set_tracking_uri(cfg.mlflow_config.MLFLOW_TRACKING_URI)
    # tracking_uri = mlflow.get_tracking_uri()
    experiment_id = mlflow.create_experiment(
        "Social NLP Experiments" + datetime.now().strftime("%m-%d %H:%M:%S"),
        artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
        tags={"version": "v1", "priority": "P1"},
    )
    experiment = mlflow.get_experiment(experiment_id)
    print(f"Name: {experiment.name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Creation timestamp: {experiment.creation_time}")
    # mlflow.set_experiment(experiment_id)
    mlflow.start_run()

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
    clf = RandomForestClassifier(n_estimators=cfg.train.estim, random_state=2)
    clf.fit(X_train, y_train)

    # Save model to DVC
    # model_path = "model/iris_model.pkl"
    with open("model/iris_model.pkl", "wb") as fd:
        dump(clf, fd)

    # mlflow.log_param('timestamp', datetime.now().strftime('%m-%d %H:%M:%S'))
    mlflow.log_params(
        {
            "C": cfg.train.estim,
            "test_size": 0.2,
            "random_state": 2,
            "columns": preprocess_cfg.columns,
        }
    )

    # Log git commit id
    commit_id = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode("utf-8")
    )
    mlflow.log_param("git_commit_id", commit_id)

    cm = confusion_matrix(y_test, clf.predict(X_test))

    # Plot confusion matrix
    fig, ax = plt.subplots()
    cl = clf.classes_
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cl)
    disp.plot(ax=ax)

    # Log the confusion matrix figure
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    # Получение предсказаний вероятностей для каждого класса
    y_probabilities = clf.predict_proba(X_test)

    # Построение Precision-Recall кривой с макро-усреднением
    plt.figure(figsize=(8, 8))

    precisions = []
    recalls = []

    for class_label in range(clf.classes_.shape[0]):
        precision, recall = custom_precision_recall(
            (y_test == class_label).astype(int), y_probabilities
        )
        macro_precision, macro_recall = custom_macro_average(precision, recall)
        precisions.extend(precision)
        recalls.extend(recall)

        plt.plot(recall, precision, label=f"Class {class_label}")

    # Макро-усреднение Precision и Recall
    macro_precision, macro_recall = custom_macro_average(precisions, recalls)

    plt.plot(
        macro_recall,
        macro_precision,
        label="Macro-Averaged Precision-Recall Curve",
        color="black",
        linestyle="--",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Macro-Averaged Precision-Recall Curve with Custom Metrics")
    plt.legend()

    thresholds = np.linspace(0, 1, 100)
    macro_auc = custom_macro_average_roc_auc(y_test, y_probabilities)

    plt.figure(figsize=(8, 8))

    plt.plot(
        thresholds,
        macro_auc,
        label="Macro-Averaged ROC AUC Curve",
        color="black",
        linestyle="--",
    )

    plt.xlabel("Threshold")
    plt.ylabel("ROC AUC")
    plt.title("Macro-Averaged ROC AUC Curve with Custom Metrics")
    plt.legend()

    # Логирование метрик в MLflow
    mlflow.log_metrics(
        {
            "macro_roc_auc_mean": np.mean(macro_auc),
        }
    )

    # Сохранение изображения и логирование в MLflow
    plt.savefig("macro_roc_auc_curve.png")
    mlflow.log_artifact("macro_roc_auc_curve.png")

    # Логирование метрик в MLflow
    mlflow.log_metrics(
        {
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
        }
    )

    # Сохранение изображения и логирование в MLflow
    plt.savefig("precision_recall_curve.png")
    mlflow.log_artifact("precision_recall_curve.png")

    # Log accuracy
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)
    mlflow.end_run()


if __name__ == "__main__":
    train()
