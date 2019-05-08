from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = r"../data/"
DATA_FILE = "data_processed.csv"

df = pd.read_csv(DATA_PATH + DATA_FILE)

y = df['label']
X = df.drop(columns=['label'])

# name, classifier, best_smote, weight (for ensembling)
models = [
    ("Ada", AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200), 0.025, 1),
    ("rf", RandomForestClassifier(n_estimators=200, n_jobs=-1), 0.015, 1),
    ("lr", LogisticRegression(solver='lbfgs', n_jobs=-1), 0.5, 1),
    ("kNN", KNeighborsClassifier(n_neighbors=7, n_jobs=-1), 0.015, 1),
    ("NB", GaussianNB(), 0.015, 1),
]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
# print(f"Before SMOTE on train: {Counter(y_train)}")
# print(f"Before SMOTE on test: {Counter(y_test)}")
#
# trained_clsfs = []
# for model_name, clsf, best_smote in models:
#     print(f"\nTraining {model_name}, smote: {best_smote}")
#     smote = SMOTE(sampling_strategy=best_smote)
#     X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
#     print(f"After SMOTE on train: {Counter(y_train_smote)}")
#     print(f"After SMOTE on test: {Counter(y_test)}")
#     trained = clsf.fit(X_train_smote, y_train_smote)
#     trained_clsfs.append(trained)
#
# ens_hard = EnsembleVoteClassifier(clfs=trained_clsfs, weights=[1 for _ in trained_clsfs], refit=False, voting="hard")
# ens_soft = EnsembleVoteClassifier(clfs=trained_clsfs, weights=[1 for _ in trained_clsfs], refit=False, voting="soft")
#
# print("hard")
# model = ens_hard.fit(X_train, y_train)
# y_pred = ens_hard.predict(X_test)
# print(metrics.f1_score(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
#
# print("soft")
# model = ens_soft.fit(X_train, y_train)
# y_pred = ens_soft.predict(X_test)
# print(metrics.f1_score(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))


# With crossval
skf = StratifiedKFold(n_splits=10, shuffle=True)
conf_matrices = {}

print("Start crossvalidation! Cross your fingers...")
for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"\n\nFold {idx+1}")
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    trained_clsfs = []
    for model_name, clsf, best_smote, weight in models:
        print(f"Training {model_name}, smote: {best_smote}")
        smote = SMOTE(sampling_strategy=best_smote)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        #print(f"After SMOTE on train: {Counter(y_train_smote)}")
        #print(f"After SMOTE on test: {Counter(y_test)}")
        trained = clsf.fit(X_train_smote, y_train_smote)
        trained_clsfs.append((trained, weight))

    ens_hard_no_refit = EnsembleVoteClassifier(clfs=[clsf for clsf, _ in trained_clsfs], weights=[weight for _, weight in trained_clsfs],
                                                refit=False, voting="hard")
    ens_soft_no_refit = EnsembleVoteClassifier(clfs=[clsf for clsf, _ in trained_clsfs], weights=[weight for _, weight in trained_clsfs],
                                                refit=False, voting="soft")
    # ens_hard = EnsembleVoteClassifier(clfs=[clsf for clsf, _ in trained_clsfs], weights=[weight for _, weight in trained_clsfs], refit=True,
    #                                   voting="hard")
    # ens_soft = EnsembleVoteClassifier(clfs=[clsf for clsf, _ in trained_clsfs], weights=[weight for _, weight in trained_clsfs], refit=True,
    #                                   voting="soft")

    for ensemble, name in zip([ens_hard_no_refit, ens_soft_no_refit],
                              ["Hard", "Soft"]):
        print(f"\nTraining {name} ensemble")
        fitted = ensemble.fit(X_train, y_train)
        y_pred = fitted.predict(X_test)
        conf_mat = metrics.confusion_matrix(y_test, y_pred)
        conf_matrices[name] = conf_matrices.get(name, np.array([[0, 0], [0, 0]])) + conf_mat
        print(conf_matrices[name])


print("\n\n------------------ FINAL RESULTS ------------------")
for name in conf_matrices.keys():
    print(name)
    tn, fp, fn, tp = conf_matrices[name].ravel()
    print(f"Precision: {tp/(tp+fp)}")
    print(f"Recall: {tp/(tp+fn)}")
