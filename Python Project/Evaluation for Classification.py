import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler


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

# Accuracy of Support Vector Machine classifier
from sklearn.svm import SVC

# svm = SVC(kernel='rbf', C=1).fit(X_train, y_train)
# print(svm.score(X_test, y_test))
#
#
# svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
# print(svm.score(X_test, y_test))


# Decision functions
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
y_scores_lr = lr.fit(X_train, y_train).decision_function(X_test)
y_score_list = list(zip(y_test[0:20], y_scores_lr[0:20]))
# print(y_score_list)  # show the decision_function scores for first 20 instances

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
y_proba_lr = lr.fit(X_train, y_train).predict_proba(X_test)
y_proba_list = list(zip(y_test[0:20], y_proba_lr[0:20,1]))
# print(y_proba_list)  # show the probability of positive class for first 20 instances


# Precision-recall curves
# from sklearn.metrics import precision_recall_curve
#
# precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)
# closest_zero = np.argmin(np.abs(thresholds))
# closest_zero_p = precision[closest_zero]
# closest_zero_r = recall[closest_zero]
#
# plt.figure()
# plt.xlim([0.0, 1.01])
# plt.ylim([0.0, 1.01])
# plt.plot(precision, recall, label='Precision-Recall Curve')
# plt.plot(closest_zero_p, closest_zero_r, 'o', markersize=12, fillstyle = 'none', c='r', mew=3)
# plt.xlabel('Precision', fontsize=16)
# plt.ylabel('Recall', fontsize=16)
# plt.axes().set_aspect('equal')
# plt.show()

# ROC curves, Area-Under-Curve (AUC)
from sklearn.metrics import roc_curve, auc

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

y_score_lr = lr.fit(X_train, y_train).decision_function(X_test)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()

# X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
#
# plt.figure()
# plt.xlim([-0.01, 1.00])
# plt.ylim([-0.01, 1.01])
# for g in [0.01, 0.1, 0.20, 1]:
#     svm = SVC(gamma=g).fit(X_train, y_train)
#     y_score_svm = svm.decision_function(X_test)
#     fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
#     roc_auc_svm = auc(fpr_svm, tpr_svm)
#     accuracy_svm = svm.score(X_test, y_test)
#     print("gamma = {:.2f}  accuracy = {:.2f}   AUC = {:.2f}".format(g, accuracy_svm,
#                                                                     roc_auc_svm))
#     plt.plot(fpr_svm, tpr_svm, lw=3, alpha=0.7,
#              label='SVM (gamma = {:0.2f}, area = {:0.2f})'.format(g, roc_auc_svm))
#
# plt.xlabel('False Positive Rate', fontsize=16)
# plt.ylabel('True Positive Rate (Recall)', fontsize=16)
# plt.plot([0, 1], [0, 1], color='k', lw=0.5, linestyle='--')
# plt.legend(loc="lower right", fontsize=11)
# plt.title('ROC curve: (1-of-10 digits classifier)', fontsize=16)
# plt.axes().set_aspect('equal')
# plt.show()