
import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score

class LogisticRegressor:

    def __init__(self, number_iterations, learning_rate):
        self.number_iterations = number_iterations
        self.learning_rate = learning_rate

    def fit(self, X_train, y_train, threshold):
        n,d = X_train.shape
        weights = np.zeros(d)
        b = 0

        self.accuracy=[]


        for _ in range(self.number_iterations):
            # first we find y_pred
            y_pred = self.sigmoid(np.dot(X_train, weights) + b)

            # append number of correct predictions for tracking model's improvement over time

            y_pred_class = np.array([0 if i <= threshold else 1 for i in y_pred])
            correct_prediction_count = accuracy_score(y_train, y_pred_class, normalize = False)
            missclasified_index = np.where(y_train != y_pred_class, 1, 0)
            misslasified_data = y_train[[missclasified for missclasified in missclasified_index if missclasified==1]]
            self.accuracy.append(correct_prediction_count)


            # optimize weights based on derivative of cost function and weights
            dw = 1 / n * np.dot(X_train.T, (y_pred - y_train))  # d dimensional vector
            db = 1 / n * np.sum(y_pred - y_train)

            # update weights and intercept based on learning rate and derivatives
            weights = weights - self.learning_rate * dw
            b = b - self.learning_rate * db

        self.weights = weights
        self.intercept = b


    def predict(self, X_test, threshold):
        y_pred_sigmoid_values = self.sigmoid(np.dot(X_test, self.weights) + self.intercept)

        y_pred = [0 if i <= threshold else 1 for i in y_pred_sigmoid_values]

        return y_pred

    def sigmoid(self, predicted_values):
        sigmoid_values = np.array([1/(1 + math.exp(-x)) for x in predicted_values])
        return sigmoid_values


