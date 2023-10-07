if __name__ == "__main__":
    from pickle import dump

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
    names = iris["target_names"]
    feature_names = iris["feature_names"]

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    svn = SVC()
    svn.fit(X_train, y_train)

    with open("./iris_pickle.pkl", "wb") as file:
        dump(svn, file)
