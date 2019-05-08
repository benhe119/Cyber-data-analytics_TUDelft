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
import math
DATA_PATH = r"../data/"
DATA_FILE = "data_processed_new.csv"

df = pd.read_csv(DATA_PATH + DATA_FILE)

y = df['label']
X = df.drop(columns=['label'])

# name, classifier, best_smote, weight (for ensembling)
k_nn = 5
models = [
    ("Ada", AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200), 0.025, 1),
    ("rf", RandomForestClassifier(n_estimators=200, n_jobs=-1), 0.015, 1),
    ("lr", LogisticRegression(solver='liblinear'), 0.15, 1),
    (f"kNN_{k_nn}", KNeighborsClassifier(n_neighbors=k_nn, n_jobs=-1), 0.025, 1),
    ("NB", GaussianNB(), 0.015, 1),
]

TEST_ENSEMBLE = True
if TEST_ENSEMBLE:
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
            trained = clsf.fit(X_train_smote, y_train_smote)
            trained_clsfs.append((trained, weight))

        ens_hard_no_refit = EnsembleVoteClassifier(clfs=[clsf for clsf, _ in trained_clsfs], weights=[weight for _, weight in trained_clsfs],
                                                    refit=False, voting="hard")
        ens_soft_no_refit = EnsembleVoteClassifier(clfs=[clsf for clsf, _ in trained_clsfs], weights=[weight for _, weight in trained_clsfs],
                                                    refit=False, voting="soft")

        for ensemble, name in zip([ens_hard_no_refit, ens_soft_no_refit],
                                  ["Hard", "Soft"]):
            print(f"\nTraining {name} ensemble")
            fitted = ensemble.fit(X_train, y_train)
            y_pred = fitted.predict(X_test)
            conf_mat = metrics.confusion_matrix(y_test, y_pred)
            conf_matrices[name] = conf_matrices.get(name, np.array([[0, 0], [0, 0]])) + conf_mat
            print(conf_matrices[name])

    for name in conf_matrices.keys():
        print(name)
        conf_mat = conf_matrices[name]
        tn, fp, fn, tp = conf_mat.ravel()
        print(conf_mat)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F0.5: {( (1+math.pow(0.5,2))*precision*recall )/( math.pow(0.5,2) * precision + recall )}")

else:
    print("Testing individual baselines")
    # test individual models as baseline
    for model_name, clsf, best_smote, _ in models:
        print(f"\n\nTraining {model_name}, smote: {best_smote}")

        conf_mat = np.array([[0, 0], [0, 0]])
        print("Start crossvalidation!")

        skf = StratifiedKFold(n_splits=10, shuffle=True)
        for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"Fold {idx + 1}")

            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            smote = SMOTE(sampling_strategy=best_smote)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

            trained = clsf.fit(X_train_smote, y_train_smote)

            y_pred = trained.predict(X_test)
            conf_mat += metrics.confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = conf_mat.ravel()
        print(conf_mat)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F0.5: {( (1+math.pow(0.5,2))*precision*recall )/( math.pow(0.5,2) * precision + recall )}")

