if __name__ == "__main__":
    from pickle import load

    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score, f1_score, precision_score
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
    names = iris["target_names"]
    feature_names = iris["feature_names"]

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    # Support vector machine algorithm
    with open("./iris_pickle.pkl", "rb") as file:
        svn = load(file)

    # Predict from the test dataset
    predictions = svn.predict(X_test)
    # Calculate the accuracy

    with open("./results.csv", "w") as file:
        file.write("x, y_true, y_pred\n")
        for X_i, y_i, y_pred_i in zip(X_test, y_test, predictions):
            s = str(X_i) + "," + str(y_i) + "," + str(y_pred_i) + "\n"
            file.write(s)

    print("accuracy: ", accuracy_score(y_test, predictions))
    print("f1: ", f1_score(y_test, predictions, average="micro"))
    print("precision: ", precision_score(y_test, predictions, average="micro"))
