from sklearn import svm

if __name__ == "__main__":
    X = "input features"
    Y = "labels"
    lin_clf = svm.LinearSVC()
    lin_clf.fit(X, Y)
    print(lin_clf.predict("Test data"))