import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


DATA_PATH = r"../data/"
DATA_FILE = "data_processed.csv"

df = pd.read_csv(DATA_PATH + DATA_FILE)

#Let's split into features and labels
y = df['label']
X = df.drop(columns = ['label'])

clf = DecisionTreeClassifier()
skf = StratifiedKFold(n_splits=10)

# print('-----NO SMOTE-----')
# conf_mat = np.array([[0,0], [0,0]])
# for train_index, test_index in skf.split(X, y):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X.loc[train_index], X.loc[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     clf = clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#
#     conf_mat += metrics.confusion_matrix(y_test, y_pred)
#
# print(conf_mat)


print('-----SMOTE-----')
max_depths = np.linspace(1, 20, 10, endpoint=True)
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)


smt = SMOTE()
conf_mat_smote = np.array([[0,0], [0,0]])

# for max_depth in max_depths:
#     clf = DecisionTreeClassifier(max_depth=max_depth)
#     conf_mat_smote = np.array([[0, 0], [0, 0]])
#     for train_index, test_index in skf.split(X, y):
#         #print("TRAIN:", train_index, "TEST:", test_index)
#         X_train, X_test = X.loc[train_index], X.loc[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         X_smote, y_smote = smt.fit_sample(X_train, y_train)
#
#         clf = clf.fit(X_smote, y_smote)
#         y_pred = clf.predict(X_test)
#
#         conf_mat_smote += metrics.confusion_matrix(y_test, y_pred)
#     print("\n", int(max_depth))
#     print(conf_mat_smote)

n_neigh = [3,5,10]
for neigh in n_neigh:
    conf_mat_smote = np.array([[0, 0], [0, 0]])
    clf = KNeighborsClassifier(n_neighbors=neigh)

    for train_index, test_index in skf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_smote, y_smote = smt.fit_sample(X_train, y_train)

        clf = clf.fit(X_smote, y_smote)
        y_pred = clf.predict(X_test)

        conf_mat_smote += metrics.confusion_matrix(y_test, y_pred)
    print("\n", neigh)
    print(conf_mat_smote)
