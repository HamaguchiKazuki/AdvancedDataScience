import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from pandas.plotting import scatter_matrix
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import BayesianRidge


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

class CurveFittingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_order=1):
        self.n_order = n_order

    def fit(self, X, y):
        pass
        
class PositiveOnlyClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        self.y_ = y
        return self
    
    def _one_side_predict(self):
        positive_num = (self.y_ > 0).sum(axis=Axis.col)
        negative_num = (self.y_ < 0).sum(axis=Axis.col)
        if positive_num.at[0] > negative_num.at[0]:
            return 1
        else:
            return -1

    def predict(self, X):
        return np.full_like(self.y_, self._one_side_predict())

train_x = pd.read_csv('data/x-tra.csv', header=None, names=[f"X{i}" for i in range(30)])
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
print(f"Correct count: {correct_count}/{len(train_y.values[0])} = {(len(train_y.T) - correct_count) / len(train_y.values[0])}")


## X13, X19, X28を抽出して学習データとする
train_x_important_feature = pd.concat([train_x.X13, train_x.X19, train_x.X28], axis=Axis.col)


# 各foldのスコアを保存するリスト
score_accuracy = []
score_logloss = []
miss_num = []
## 交差検証
from sklearn.model_selection import KFold
kf = KFold(n_splits=4, shuffle=True, random_state=42)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x_important_feature.iloc[tr_idx], train_x_important_feature.iloc[va_idx]
    tr_y, va_y = train_y.T.iloc[tr_idx], train_y.T.iloc[va_idx]

    model = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True, alpha_init=1., lambda_init=1)
    model.fit(tr_x, tr_y)

    va_pred = model.predict(va_x)
    # va_pred_serise = va_pred[:, 1]

    one_and_minus_one = [1 if x > 0.0 else -1 for x in va_pred]
    
    accuracy = accuracy_score(va_y, one_and_minus_one)
    logloss = log_loss(va_y, one_and_minus_one)

    score_accuracy.append(accuracy)
    miss_num.append(len(va_y) - (va_y.T.values == one_and_minus_one).sum())
    score_logloss.append(logloss)

logloss = np.mean(score_logloss)
accuracy = np.mean(score_accuracy)
miss_num_sum = np.sum(miss_num)

print(f"logloss:{logloss:.4f}, accuracy:{accuracy:.4f}")

model.fit(train_x_important_feature, train_y.T)
train_pred = model.predict(train_x_important_feature)
train_one_and_minus_one = [1 if x > 0.0 else -1 for x in train_pred]

submission = pd.DataFrame(train_one_and_minus_one)
submission.T.to_csv('train_submission.csv', sep=",", index=False, header=False)

test_x = pd.read_csv('data/x-test.csv', header=None, names=[f"X{i}" for i in range(30)])
model.fit(train_x_important_feature, train_y.T)

test_x_important_feature = pd.concat([test_x.X13, test_x.X19, test_x.X28], axis=Axis.col)


test_pred = model.predict(test_x_important_feature)
test_one_and_minus_one = [1 if x > 0.0 else -1 for x in test_pred]

submission = pd.DataFrame(test_one_and_minus_one)
submission.T.to_csv('test_submission.csv', sep=",", index=False, header=False)



# scatter_matrix(train_x.loc[:,'X0':'X5'], c=train_y, figsize=(15,15), marker="o", hist_kwds={'bins':20}, cmap="PiYG")
# scatter_matrix(train_x.loc[:,'X6':'X10'], c=train_y, figsize=(15,15), marker="o", hist_kwds={'bins':20}, cmap="PiYG")
# scatter_matrix(train_x.loc[:,'X11':'X15'], c=train_y, figsize=(15,15), marker="o", hist_kwds={'bins':20}, cmap="PiYG")
# scatter_matrix(train_x.loc[:,'X16':'X20'], c=train_y, figsize=(15,15), marker="o", hist_kwds={'bins':20}, cmap="PiYG")
# scatter_matrix(train_x.loc[:,'X21':'X25'], c=train_y, figsize=(15,15), marker="o", hist_kwds={'bins':20}, cmap="PiYG")
# scatter_matrix(train_x.loc[:,'X26':'X29'], c=train_y, figsize=(15,15), marker="o", hist_kwds={'bins':20}, cmap="PiYG")
# train_x.plot.scatter(x="X13", y="X19", c=train_y, cmap="PiYG")
# train_x.plot.scatter(x="X13", y="X28", c=train_y, cmap="PiYG")
# train_x.plot.scatter(x="X19", y="X28", c=train_y, cmap="PiYG")