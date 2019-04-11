from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

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


clf = SVC(kernel='linear')
grid_values = {'class_weight': ['balanced', {1: 2}, {1: 3}, {1: 4}, {1: 5}, {1: 10}, {1: 20}, {1: 50}]}
plt.figure(figsize=(9, 6))
for i, eval_metric in enumerate(('precision', 'recall', 'f1', 'roc_auc')):
    grid_clf_custom = GridSearchCV(clf, param_grid=grid_values, scoring=eval_metric)
    grid_clf_custom.fit(X_train, y_train)
    print('Grid best parameter (max. {0}): {1}'
          .format(eval_metric, grid_clf_custom.best_params_))
    print('Grid best score ({0}): {1}'
          .format(eval_metric, grid_clf_custom.best_score_))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plot_class_regions_for_classifier_subplot(grid_clf_custom, X_test, y_test, None)

    plt.title(eval_metric + '-oriented SVC')
plt.tight_layout()
plt.show()