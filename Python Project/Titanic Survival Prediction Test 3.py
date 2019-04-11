import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")

y = train['Survived']

train.drop(['Survived'], axis=1, inplace=True)

train.drop(['Cabin'], axis=1, inplace=True)
train['Embarked'].fillna('S', inplace=True)  # fill nan in Embarked with top freq 'S'
train['Embarked'] = pd.factorize(train['Embarked'])[0]
train['Ticket'] = pd.factorize(train['Ticket'])[0]
train['Sex'] = pd.factorize(train['Sex'])[0]
train.drop(['PassengerId'], axis=1, inplace=True)

train['Title'] = train['Name'].str.extract('([A-Za-z]+)\.', expand=False)
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')
train['Title'] = train['Title'].replace('Rev', 'Mr')

train['Title'] = train['Title'].replace(['Capt', 'Col', 'Countess', 'Don',
                                         'Dr', 'Major', 'Sir', 'Jonkheer', 'Lady'], 'Noble')

X_age = train.dropna(axis=0)
X_age_class = X_age.groupby('Pclass').agg({'Age': np.average})
X_age_title = X_age.groupby('Title').agg({'Age': np.average})
print(X_age.groupby('Pclass').agg({'Age': np.average}))
print(X_age.groupby('Title').agg({'Age': np.average}))

train['Age'].fillna(0, inplace=True)
train.drop('Name', axis=1, inplace=True)

X_train_M_age = train[train['Age'] == 0]

X_train_M_age.set_value(X_train_M_age[X_train_M_age['Title'] == 'Master'].index,
                        'Age', X_age_title.loc['Master'].item())
X_train_M_age.set_value(X_train_M_age[X_train_M_age['Title'] == 'Noble'].index,
                        'Age', X_age_title.loc['Noble'].item())

for cla in X_train_M_age['Pclass']:
    for title in ['Miss', 'Mr', 'Mrs']:
        X_train_M_age.set_value(X_train_M_age[(X_train_M_age['Pclass'] == cla) & (X_train_M_age['Title'] == title)].index, 'Age',
         (X_age_class.loc[cla].item() + X_age_title.loc[title].item()) / 2)

print(X_train_M_age)

train.set_value(train[train['Age'] == 0].index, 'Age', X_train_M_age['Age'].values)  # set missing ages completed
print(train)

train.to_csv('train_age.csv', index=False)
