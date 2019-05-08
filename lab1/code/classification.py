from collections import Counter

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = r"../data/"
DATA_FILE = "data_processed.csv"

df = pd.read_csv(DATA_PATH + DATA_FILE)

# Let's split into features and labels
y = df['label']
X = df.drop(columns=['label'])

# We first split in train and test set -> we only have to apply SMOTE on train set
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3)

# Now we split on train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)

classifiers = [
    ("Ada", AdaBoostClassifier(n_estimators=50)),
    ("rf", RandomForestClassifier(n_estimators=50, n_jobs=-1)),
    ("lr", LogisticRegression(n_jobs=-1)),
    ("kNN", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ("NB", GaussianNB()),

]

smote_values = [0, 0.015, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0]
best_smotes = []
for clsf_name, clsf in classifiers:
    print(f"\n\nUsing '{clsf_name}'")
    acc = []
    recall = []
    precision = []
    f05_score = []
    f1_score = []
    roc_auc = []
    conf_mat = []
    for smote_perc in smote_values:
        print(f'\n--- SMOTE PERCENTAGE: {smote_perc} ---')
        #print(f"Before SMOTE on train: {Counter(y_train)}")
        #print(f"Before SMOTE on validation: {Counter(y_val)}")

        smt = SMOTE(sampling_strategy=smote_perc) if smote_perc else None
        X_smote, y_smote = smt.fit_sample(X_train, y_train) if smote_perc else (X_train, y_train)

        #print(f"After SMOTE on train: {Counter(y_smote)}")
        #print(f"After SMOTE on validation: {Counter(y_val)}")

        clsf.fit(X_smote, y_smote)
        y_pred = clsf.predict(X_val)

        acc.append(metrics.accuracy_score(y_val, y_pred))
        recall.append(metrics.recall_score(y_val, y_pred))
        precision.append(metrics.precision_score(y_val, y_pred))
        f05_score.append(metrics.fbeta_score(y_val, y_pred, beta=0.5))
        f1_score.append(metrics.f1_score(y_val, y_pred))
        roc_auc.append(metrics.roc_auc_score(y_val, y_pred))
        conf_mat.append(metrics.confusion_matrix(y_val, y_pred))

    fig = plt.figure()
    fig.suptitle(f" Scores for {clsf_name}", y=1)

    plt.subplot(2, 2, 1)
    plt.plot(smote_values, precision, color='green', label='precision')
    plt.xlabel("Smote percentage")
    plt.ylabel("Precision")

    plt.subplot(2, 2, 2)
    plt.plot(smote_values, recall, color='black', label='recall')
    plt.xlabel("Smote percentage")
    plt.ylabel("Recall")

    plt.subplot(2, 2, 3)
    plt.plot(smote_values, f1_score, color='blue', label='F1')
    plt.xlabel("Smote percentage")
    plt.ylabel("F1")


    plt.subplot(2, 2, 4)
    plt.plot(smote_values, f05_score, color='red', label='F0.5')
    plt.xlabel("Smote percentage")
    plt.ylabel("F0.5")

    plt.show()

    print("\n")
    for metric_name, mets in zip(["Precision", "Recall", "F1", "F0.5", ], [precision, recall, f1_score, f05_score]):
        max_ind = np.argmax(mets)
        print(f"Max '{metric_name}' --> {mets[max_ind]} (SMOTE perc: {smote_values[max_ind]})")
        print(conf_mat[max_ind])
        print("\n")

        if metric_name == "F0.5":
            best_smotes.append(smote_values[max_ind])


print(best_smotes)


for (clsf_name, clsf), best_smote in zip(classifiers, best_smotes):
    print(f"{clsf_name} --> {best_smote}")

    fpr_smoted = []
    tpr_smoted = []
    if best_smote:
        smt = SMOTE(sampling_strategy=best_smote)
        X_smote, y_smote = smt.fit_sample(X_train_val, y_train_val)
        clsf.fit(X_smote, y_smote)
        y_pred = clsf.predict(X_test)

        probs_smoted = clsf.predict_proba(X_test)
        probs_smoted = probs_smoted[:, 1]
        fpr_smoted, tpr_smoted, _ = roc_curve(y_test, probs_smoted)
        plt.plot(fpr_smoted, tpr_smoted, color='red', label='SMOTEd')

    clsf.fit(X_train_val, y_train_val)
    y_pred = clsf.predict(X_test)
    probs_smoted = clsf.predict_proba(X_test)
    probs_smoted = probs_smoted[:, 1]
    fpr_unsmoted, tpr_unsmoted, _ = roc_curve(y_test, probs_smoted)
    plt.plot(fpr_unsmoted, tpr_unsmoted, color='blue', label='UNSMOTEd')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {clsf_name} - best SMOTE: {best_smote}')
    plt.legend()
    plt.show()


# #Apply SMOTE
# print(f"Before SMOTE: Fraud cases train: {len(y_train[y_train==1])} and non-fraud cases train: {len(y_train[y_train==0])}")
# print(f"Before SMOTE: Fraud cases test: {len(y_test[y_test==1])} and non-fraud cases train: {len(y_test[y_test==0])}")
#
# brd_smote = BorderlineSMOTE()
# smt = SMOTE()
# X_smote, y_smote = smt.fit_sample(X_train, y_train)
#
# print(f"After SMOTE: Fraud cases train: {len(y_smote[y_smote == 1])} and non-fraud cases: {len(y_smote[y_smote ==0 ])}")
# print(f"After SMOTE: Fraud cases test: {len(y_test[y_test==1])} and non-fraud cases train: {len(y_test[y_test==0])}")
#
#
# classifiers = [
#     ("Knn", KNeighborsClassifier(n_neighbors=7)),
#     ("Random Forest", RandomForestClassifier(n_estimators=150)),
#     ("Logistic Regression", LogisticRegression()),
#     #("SVM", svm.SVC())
# ]
#
# for model_name, model in classifiers:
#     print(f'---------- {model_name} BEFORE SMOTE----------')
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     probs = model.predict_proba(X_test)
#     probs = probs[:, 1]  # only positive class
#     fpr_unsmoted, tpr_unsmoted, _ = roc_curve(y_test, probs)
#     precision_unsmoted, recall_unsmoted, _ = precision_recall_curve(y_test, probs)
#
#     print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
#     print("Recall:", metrics.recall_score(y_test, y_pred))
#     print("Precision:", metrics.precision_score(y_test, y_pred))
#     print("F1: ", metrics.f1_score(y_test, y_pred))
#     print("ROC AUC Score:", metrics.roc_auc_score(y_test, y_pred))
#     print(metrics.confusion_matrix(y_test, y_pred))
#
    # print(f'---------- {model_name} AFTER SMOTE----------')
    # model.fit(X_smote, y_smote)
    # y_pred = model.predict(X_test)
    #
    # probs_smoted = model.predict_proba(X_test)
    # probs_smoted = probs_smoted[:, 1]
    # fpr_smoted, tpr_smoted, _ = roc_curve(y_test, probs_smoted)
    # precision_smoted, recall_smoted, _ = precision_recall_curve(y_test, probs_smoted)
    #
    # # Model Accuracy, how often is the classifier correct?
    # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # print("Recall:", metrics.recall_score(y_test, y_pred))
    # print("Precision:", metrics.precision_score(y_test, y_pred))
    # print("F1: ", metrics.f1_score(y_test, y_pred))
    # print("ROC AUC Score:", metrics.roc_auc_score(y_test, y_pred))
    # print(metrics.confusion_matrix(y_test, y_pred))
    #
    # plt.plot(fpr_unsmoted, tpr_unsmoted, color='blue', label='UNSMOTEd')
    # plt.plot(fpr_smoted, tpr_smoted, color='red', label='SMOTEd')
    # plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(f'ROC Curves - {model_name}')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(recall_unsmoted, precision_unsmoted, color='blue', label='UNSMOTEd')
    # plt.plot(recall_smoted, precision_smoted, color='red', label='SMOTEd')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(f'PR Curves - {model_name}')
    # plt.legend()
    # plt.show()
