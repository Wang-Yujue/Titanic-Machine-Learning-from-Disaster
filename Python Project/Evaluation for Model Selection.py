from sklearn.model_selection import train_test_split
import pandas as pd
# Model selection using evaluation metrics
# Cross-validation example
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

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

# clf = SVC(kernel='linear', C=1)
#
# # accuracy is the default scoring metric
# print('Cross-validation (accuracy)', cross_val_score(clf, X, y, cv=5))
# # use AUC as scoring metric
# print('Cross-validation (AUC)', cross_val_score(clf, X, y, cv=5, scoring='roc_auc'))
# # use recall as scoring metric
# print('Cross-validation (recall)', cross_val_score(clf, X, y, cv=5, scoring='recall'))

# Grid search example
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = SVC(kernel='rbf')
grid_values = {'gamma': [0.0001, 0.001, 0.01, 0.05, 0.1, 1, 10, 100]}

# default metric to optimize over grid parameters: accuracy
grid_clf_acc = GridSearchCV(clf, param_grid=grid_values)
grid_clf_acc.fit(X_train, y_train)
y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test)

print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc.best_score_)

# alternative metric to optimize over grid parameters: AUC
grid_clf_auc = GridSearchCV(clf, param_grid=grid_values, scoring='roc_auc')
grid_clf_auc.fit(X_train, y_train)
y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test)

print('Test set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)

# Evaluation metrics supported for model selection
# from sklearn.metrics.scorer import SCORERS
#
# print(sorted(list(SCORERS.keys())))