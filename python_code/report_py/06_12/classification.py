import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array


class Axis():
    row = 0
    col = 1


class TemplateClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]

class PositiveOnlyClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        self.y_ = y
        return self
    
    def _abs_positive(self, X):
        return np.array[(1)]

    def predict(self, X):
        return np.ones_like(self.y_)

train_x = pd.read_csv('data/x-tra.csv', header=None)
train_y = pd.read_csv('data/y-tra.csv', header=None, nrows=1)

positive_bool = train_y > 0
negative_bool = train_y < 0

positive_num = positive_bool.sum(axis=Axis.col)
negative_num = negative_bool.sum(axis=Axis.col)
print(f"Ratio positive:negative, {positive_num[0]}:{negative_num[0]}")

model = PositiveOnlyClassifier()
model.fit(train_x, train_y)
positive_only_y = model.predict(train_x)
correct_count = (positive_only_y[0] == train_y.values[0]).sum()
print(f"Correct count: {correct_count}/{len(train_y.values[0])} = {correct_count/len(train_y.values[0])}")
