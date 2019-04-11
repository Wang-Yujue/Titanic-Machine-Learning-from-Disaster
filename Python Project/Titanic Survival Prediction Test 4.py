import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from matplotlib import pyplot as plt


train = pd.read_csv("train.csv")
y = train['Survived']

train = pd.read_csv("train_age.csv")
train['Title'] = pd.factorize(train['Title'])[0] + 1
train['Embarked'] = train['Embarked'] + 1
train['Ticket'] = train['Ticket'] + 1
train['Sex'] = train['Sex'] + 1
# train['Company'] = train['SibSp'] + train['Parch']


# X = train.drop(['Ticket', 'SibSp', 'Parch'], axis=1)
X = train.drop(['Ticket'], axis=1)


X_scaled = MinMaxScaler().fit_transform(X)

# clf = SVC()
clf = RandomForestClassifier(n_estimators=26, random_state=0)
# clf = DecisionTreeClassifier(random_state=0)
# clf = KNeighborsClassifier()
# clf = GaussianNB()
# # clf = LogisticRegression()
clf = MLPClassifier(hidden_layer_sizes=[100, 100], solver='adam', max_iter=300, random_state=1)
# cv_scores = cross_val_score(clf, X, y, cv=5)
cv_scores = cross_val_score(clf, X_scaled, y, cv=5)

print('Cross-validation scores (5-fold):', cv_scores)
print('Mean cross-validation score (5-fold): {:.3f}'.format(np.mean(cv_scores)))

# validation curve
# param_range = ['lbfgs', 'sgd', 'adam']
# train_scores, test_scores = \
#     validation_curve(MLPClassifier(hidden_layer_sizes=[100, 100], alpha=0.01, max_iter=0.01, random_state=0), X_scaled, y,
#                                             param_name='solver',
#                                             param_range=param_range, cv=5)
#
# plt.plot(param_range, np.mean(test_scores, axis=1))
# plt.plot(param_range, np.mean(train_scores, axis=1))
# plt.show()