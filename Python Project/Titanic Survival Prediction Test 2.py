import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from matplotlib import pyplot as plt


train = pd.read_csv("train.csv")

y = train['Survived']

train.drop(['Survived'], axis=1, inplace=True)

# survive_rate = len(y[y==1])/len(y)  # survive rate is 38.38%
# train['Cabin'] = pd.factorize(train['Cabin'])[0]
train.drop(['Cabin'], axis=1, inplace=True)
# train['Embarked'].fillna('S', inplace=True)  # fill nan in Embarked with top freq 'S'
# train['Embarked'] = pd.factorize(train['Embarked'])[0]
train.drop(['Embarked'], axis=1, inplace=True)
train['Ticket'] = pd.factorize(train['Ticket'])[0]
# train.drop(['Ticket'], axis=1, inplace=True)
train['Age'].fillna(train['Age'].mean(), inplace=True)
train['Sex'] = pd.factorize(train['Sex'])[0]
train.drop(['Name'], axis=1, inplace=True)
train.drop(['PassengerId'], axis=1, inplace=True)


X = train

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train_scaled = MinMaxScaler().fit_transform(X_train)
X_test_scaled = MinMaxScaler().fit_transform(X_test)

# # Decision Tree
# train_score = []
# test_score = []
# for depth in [6, 7, 8]:
#     for leaf in [4, 5, 6, 7]:
#         clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(X_train, y_train)
#         train_score.append(clf.score(X_train, y_train))
#         test_score.append(clf.score(X_test, y_test))
#
# print(train_score)
# print(test_score)
# # depth = 7, leaf = 5 gives the best performance

# clf = DecisionTreeClassifier(max_depth=7, min_samples_leaf=5, random_state=0).fit(X_train, y_train)
# train_score = clf.score(X_train, y_train)
# test_score = clf.score(X_test, y_test)
#
# print(train_score)
# print(test_score)


# Random Forest
# Normalization has little effect
# train_score = []
# test_score = []
# for n in [50, 100, 200]:
#     for feature in [3]:
#         clf = RandomForestClassifier(n_estimators=n, max_features=feature, random_state=0).fit(X_train, y_train)
#         train_score.append(clf.score(X_train, y_train))
#         test_score.append(clf.score(X_test, y_test))
#
#
# print(train_score)
# print(test_score)


clf = RandomForestClassifier(n_estimators=49, max_features='auto', random_state=0).fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)

print(train_score)
print(test_score)


# train_score = []
# test_score = []
# for this_gamma in [0.01, 1, 5]:
#     for this_C in [0.1, 1, 15, 250]:
#         clf = SVC(kernel='rbf', gamma=this_gamma,
#                   C=this_C).fit(X_train_scaled, y_train)
#         train_score.append(clf.score(X_train_scaled, y_train))
#         test_score.append(clf.score(X_test_scaled, y_test))
#
# print(train_score)
# print(test_score)


# clf = SVC(kernel='rbf', gamma=1, C=15).fit(X_train_scaled, y_train)
# train_score = clf.score(X_train_scaled, y_train)
# test_score = clf.score(X_test_scaled, y_test)
#
# print(train_score)
# print(test_score)


# KNN
# train_score = []
# test_score = []
# for n in [4, 5, 6, 7]:
#     knn = KNeighborsClassifier(n_neighbors=n).fit(X_train_scaled, y_train)
#     train_score.append(knn.score(X_train_scaled, y_train))
#     test_score.append(knn.score(X_test_scaled, y_test))
#
# print(train_score)
# print(test_score)


# clf = SVC(kernel='rbf', gamma=1, C=15)
clf = RandomForestClassifier(n_estimators=21, max_features='auto', random_state=0)
# clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=0)
cv_scores = cross_val_score(clf, X, y, cv=5)

print('Cross-validation scores (5-fold):', cv_scores)
print('Mean cross-validation score (5-fold): {:.3f}'.format(np.mean(cv_scores)))

# validation curve
# param_range = range(20, 30)
# train_scores, test_scores = \
#     validation_curve(RandomForestClassifier(max_features='auto', random_state=0), X, y,
#                                             param_name='n_estimators',
#                                             param_range=param_range, cv=5)
#
# # print(np.mean(train_scores, axis=1))
# plt.plot(param_range, np.mean(test_scores, axis=1))
# plt.show()

# param_range = range(5, 15)
# train_scores, test_scores = \
#     validation_curve(DecisionTreeClassifier(), X, y,
#                                             param_name='min_samples_leaf'
#                                             param_range=param_range, cv=5)
#
# # print(np.mean(train_scores, axis=1))
# plt.plot(param_range, np.mean(test_scores, axis=1))
# plt.show()