import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn import metrics
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, RUSBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import StratifiedKFold

DATA_PATH = r"../data/"
DATA_FILE = "data_processed.csv"

df = pd.read_csv(DATA_PATH + DATA_FILE)

y = df['label']
X = df.drop(columns=['label'])

params = {
    'n_estimators': 1,
    'max_depth': 1,
    'learning_rate': 1,
    'criterion': 'mse'
}

clf = GradientBoostingClassifier(**params)

skf = StratifiedKFold(n_splits=10)
conf_mat = np.array([[0, 0], [0, 0]])
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(idx)
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    smt = SMOTE()
    adasyn = ADASYN()
    X_smote, y_smote = smt.fit_resample(X_train, y_train)
    X_adasyn, y_adasyn =  adasyn.fit_resample(X_train, y_train)

    model = clf.fit(X_adasyn, y_adasyn)
    y_pred = model.predict(X_test)
    conf_mat += metrics.confusion_matrix(y_test, y_pred)
    print(conf_mat)

t_p = conf_mat[1][1]
f_p = conf_mat[0][1]
t_n = conf_mat[0][0]
f_n = conf_mat[1][0]

print("Nothing")
print(f"Prec: {t_p/(t_p + f_p)}")
print(f"Rec: {t_p/(t_p+f_n)}")
