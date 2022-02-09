# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from LogisticRegressor import LogisticRegressor
from sklearn.metrics import accuracy_score, confusion_matrix


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X, y = make_blobs(n_samples=1000, centers=[(1,2),(3,4)], n_features=2)

    # scatter plot to see what we made
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    # creating datasets
    X_train = X[:800,]
    y_train = y[:800,]
    X_test = X[800:,]
    y_test = y[800:,]



    # creating Logistic regressor object with threshold 0.1 for positive values
    logistic_regressor1 = LogisticRegressor(1000, 0.1)
    # train logistic regressor
    threshold_list = [0.2, 0.5, 0.7]
    for i in threshold_list:
        logistic_regressor1.fit(X_train, y_train, i)
        y_predict = logistic_regressor1.predict(X_test, i)
        print(confusion_matrix(y_test, y_predict))
        print("Out of 200, Number of correct predictions", accuracy_score(y_test, y_predict, normalize=False))


    # cost plot - error for each iteration during training
    plt.plot(range(logistic_regressor1.number_iterations), logistic_regressor1.accuracy)
    plt.title("Improvement of our training model")
    plt.xlabel("Number of iterations")
    plt.ylabel("Number of correct predictions")
    plt.show()








