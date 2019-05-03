import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn import metrics, svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB

import numpy as np

DATA_PATH = r"../data/"
DATA_FILE = "data_processed.csv"

df = pd.read_csv(DATA_PATH + DATA_FILE)

y = df['label']
X = df.drop(columns=['label'])

ada = AdaBoostClassifier(n_estimators=100)
rf = RandomForestClassifier(n_estimators=100)
one_class_svm = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
g_nb = GaussianNB()

skf = StratifiedKFold(n_splits=10)
conf_mat = np.array([[0, 0], [0, 0]])
conf_mat_smote = np.array([[0, 0], [0, 0]])
conf_mat_syn = np.array([[0, 0], [0, 0]])

for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(idx)
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    smt = SMOTE()
    adasyn = ADASYN()
    X_smote, y_smote = smt.fit_resample(X_train, y_train)
    X_adasyn, y_adasyn =  adasyn.fit_resample(X_train, y_train)

    print(f"Before SMOTE: Fraud cases train: {len(y_train[y_train == 1])} and non-fraud cases train: {len(y_train[y_train == 0])}")
    print(f"Before SMOTE: Fraud cases test: {len(y_test[y_test == 1])} and non-fraud cases train: {len(y_test[y_test == 0])}")
    print(f"After SMOTE: Fraud cases train: {len(y_smote[y_smote == 1])} and non-fraud cases: {len(y_smote[y_smote == 0])}")
    print(f"After SMOTE: Fraud cases test: {len(y_test[y_test == 1])} and non-fraud cases train: {len(y_test[y_test == 0])}")

    model = ada.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_mat += metrics.confusion_matrix(y_test, y_pred)
    print("No smote")
    print(conf_mat)

    model_smote = ada.fit(X_smote, y_smote)
    y_pred_smote = model_smote.predict(X_test)
    conf_mat_smote += metrics.confusion_matrix(y_test, y_pred_smote)
    print("With smote")
    print(conf_mat_smote)

    model_syn = ada.fit(X_adasyn, y_adasyn)
    y_pred_syn = model_syn.predict(X_test)
    conf_mat_syn += metrics.confusion_matrix(y_test, y_pred_syn)
    print("With syn")
    print(conf_mat_syn)

    # model = one_class_svm.fit(X_train)
    # y_pred = model.predict(X_test)
    # conf_mat += metrics.confusion_matrix(y_test, y_pred)

    # model = rf.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # conf_mat += metrics.confusion_matrix(y_test, y_pred)

    # model = g_nb.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # conf_mat += metrics.confusion_matrix(y_test, y_pred)
    # print(conf_mat)

t_p = conf_mat[1][1]
f_p = conf_mat[0][1]
t_n = conf_mat[0][0]
f_n = conf_mat[1][0]

print("Nothing")
print(f"Prec: {t_p/(t_p + f_p)}")
print(f"Rec: {t_p/(t_p+f_n)}")

t_p = conf_mat_smote[1][1]
f_p = conf_mat_smote[0][1]
t_n = conf_mat_smote[0][0]
f_n = conf_mat_smote[1][0]

print("smote")
print(f"Prec: {t_p/(t_p + f_p)}")
print(f"Rec: {t_p/(t_p+f_n)}")


t_p = conf_mat_syn[1][1]
f_p = conf_mat_syn[0][1]
t_n = conf_mat_syn[0][0]
f_n = conf_mat_syn[1][0]

print("syn")
print(f"Prec: {t_p/(t_p + f_p)}")
print(f"Rec: {t_p/(t_p+f_n)}")