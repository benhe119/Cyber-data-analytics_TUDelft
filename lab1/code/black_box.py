import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = r"../data/"
DATA_FILE = "data_processed.csv"

df = pd.read_csv(DATA_PATH + DATA_FILE)

y = df['label']
X = df.drop(columns=['label'])

# just one split
lr = LogisticRegression(solver='lbfgs')
rf = RandomForestClassifier(n_estimators=100,)
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)


ens_hard = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('adaboost', ada)], voting='hard')
ens_soft = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('adaboost', ada)], voting='soft')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print(f"TRAIN, NO SMOTE: positives: {len(y_train[y_train == 1])} | negatives: {len(y_train[y_train == 0])}")
print(f"TEST, NO SMOTE: positives: {len(y_test[y_test == 1])} | negatives: {len(y_test[y_test == 0])}")

smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"TRAIN, SMOTEd: positives: {len(y_train_smote[y_train_smote == 1])} | negatives: {len(y_train_smote[y_train_smote == 0])}")
print(f"TEST, SMOTEd: positives: {len(y_test[y_test == 1])} |negatives : {len(y_test[y_test == 0])}")

print("\n\n")
for clf, label in zip([lr, rf, ada, ens_hard, ens_soft], ['Logistic Regression', 'Random Forest', 'Adaboost', 'Ensemble hard', 'Ensemble soft']):
    print("\n")
    print(label)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{label} | NO SMOTE | Precision:", metrics.f1_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    clf.fit(X_train_smote, y_train_smote)
    y_pred_smote = clf.predict(X_test)
    print(f"{label} | SMOTE | Precision:", metrics.f1_score(y_test, y_pred_smote))
    print(confusion_matrix(y_test, y_pred_smote))


# With crossval
#
# skf = StratifiedKFold(n_splits=10)
# conf_mat = np.array([[0, 0], [0, 0]])
#
# for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
#     print(idx)
#     X_train, X_test = X.loc[train_index], X.loc[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     smt = SMOTE()
#     adasyn = ADASYN()
#     X_smote, y_smote = smt.fit_resample(X_train, y_train)
#     X_adasyn, y_adasyn =  adasyn.fit_resample(X_train, y_train)
#
#     print(f"Before SMOTE: Fraud cases train: {len(y_train[y_train == 1])} and non-fraud cases train: {len(y_train[y_train == 0])}")
#     print(f"Before SMOTE: Fraud cases test: {len(y_test[y_test == 1])} and non-fraud cases train: {len(y_test[y_test == 0])}")
#     print(f"After SMOTE: Fraud cases train: {len(y_smote[y_smote == 1])} and non-fraud cases: {len(y_smote[y_smote == 0])}")
#     print(f"After SMOTE: Fraud cases test: {len(y_test[y_test == 1])} and non-fraud cases train: {len(y_test[y_test == 0])}")
#
#     model = ada.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     conf_mat += metrics.confusion_matrix(y_test, y_pred)
#     print("No smote")
#     print(conf_mat)
#
#
# t_p = conf_mat[1][1]
# f_p = conf_mat[0][1]
# t_n = conf_mat[0][0]
# f_n = conf_mat[1][0]
# print(f"Prec: {t_p/(t_p + f_p)}")
# print(f"Rec: {t_p/(t_p+f_n)}")
