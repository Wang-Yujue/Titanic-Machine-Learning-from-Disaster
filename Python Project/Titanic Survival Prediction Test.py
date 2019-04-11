import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


train = pd.read_csv("train.csv")
y = train['Survived']

# survive_rate = len(y[y==1])/len(y)  # survive rate is 38.38%
train['Cabin'] = pd.factorize(train['Cabin'])[0]
train['Embarked'] = pd.factorize(train['Embarked'])[0]
train.drop(['Ticket'], axis=1, inplace=True)
train['Age'].fillna(train['Age'].mean(), inplace=True)
train['Sex'] = pd.factorize(train['Sex'])[0]
train.drop(['Name'], axis=1, inplace=True)
train.drop(['PassengerId'], axis=1, inplace=True)
train.drop(['Survived'], axis=1, inplace=True)

X = train

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# Application of SVMs to a real dataset: normalized data with feature preprocessing using minmax scaling

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # SVM
# svm = SVC(gamma=1, C=250).fit(X_train_scaled, y_train)
#
# print('Titanic dataset (normalized with MinMax scaling)')
# print('RBF-kernel SVC (with MinMax scaling) training set accuracy: {:.2f}'
#       .format(svm.score(X_train_scaled, y_train)))
# print('RBF-kernel SVC (with MinMax scaling) test set accuracy: {:.2f}'
#       .format(svm.score(X_test_scaled, y_test)))
#
#
# # Support Vector Machine with RBF kernel: using both C and gamma parameter
#
# train_score = []
# test_score = []
# for this_gamma in [0.01, 1, 5]:
#     for this_C in [0.1, 1, 15, 250]:
#         title = 'gamma = {:.2f}, C = {:.2f}'.format(this_gamma, this_C)
#         clf = SVC(kernel='rbf', gamma=this_gamma,
#                   C=this_C).fit(X_train_scaled, y_train)
#         train_score.append(clf.score(X_train_scaled, y_train))
#         test_score.append(clf.score(X_test_scaled, y_test))
#
# print(train_score)
# print(test_score)


# Random Forest
# Normalization has little effect
# train_score = []
# test_score = []
# for param in [1, 10, 50, 100]:
#     clf = RandomForestClassifier(n_estimators=param, random_state=0).fit(X_train, y_train)
#     train_score.append(clf.score(X_train, y_train))
#     test_score.append(clf.score(X_test, y_test))
#
# print(train_score)
# print(test_score)

# train_score = []
# test_score = []
# for param in [2, 3, 4]:
#     clf = RandomForestClassifier(max_features=param, random_state=0).fit(X_train_scaled, y_train)
#     train_score.append(clf.score(X_train_scaled, y_train))
#     test_score.append(clf.score(X_test_scaled, y_test))
#
# print(train_score)
# print(test_score)


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
# depth = 7, leaf = 5 gives the best performance


train_score = []
test_score = []
for depth in [5, 6, 7, 8, 9]:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_train, y_train)
        train_score.append(clf.score(X_train, y_train))
        test_score.append(clf.score(X_test, y_test))

print(train_score)
print(test_score)


train_score = []
test_score = []
for leaf in [4, 5, 6, 7, 8, 9, 10]:
        clf = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=0).fit(X_train, y_train)
        train_score.append(clf.score(X_train, y_train))
        test_score.append(clf.score(X_test, y_test))

print(train_score)
print(test_score)

