from imblearn.over_sampling import SMOTE, BorderlineSMOTE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

DATA_PATH = r"../data/"
DATA_FILE = "data_processed.csv"

df = pd.read_csv(DATA_PATH + DATA_FILE)

#Let's split into features and labels
y = df['label']
X = df.drop(columns = ['label'])

#We first split in train and test set -> we only have to apply SMOTE on train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#Apply SMOTE
print(f"Before SMOTE: Fraud cases train: {len(y_train[y_train==1])} and non-fraud cases train: {len(y_train[y_train==0])}")
print(f"Before SMOTE: Fraud cases test: {len(y_test[y_test==1])} and non-fraud cases train: {len(y_test[y_test==0])}")

brd_smote = BorderlineSMOTE()
smt = SMOTE()
X_smote, y_smote = smt.fit_sample(X_train, y_train)

print(f"After SMOTE: Fraud cases train: {len(y_smote[y_smote == 1])} and non-fraud cases: {len(y_smote[y_smote ==0 ])}")
print(f"After SMOTE: Fraud cases test: {len(y_test[y_test==1])} and non-fraud cases train: {len(y_test[y_test==0])}")


classifiers = [
    ("Knn", KNeighborsClassifier(n_neighbors=7)),
    ("Random Forest", RandomForestClassifier(n_estimators=150)),
    ("Logistic Regression", LogisticRegression()),
    #("SVM", svm.SVC())
]

for model_name, model in classifiers:
    print(f'---------- {model_name} BEFORE SMOTE----------')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    probs = model.predict_proba(X_test)
    probs = probs[:, 1]  # only positive class
    fpr_unsmoted, tpr_unsmoted, _ = roc_curve(y_test, probs)
    precision_unsmoted, recall_unsmoted, _ = precision_recall_curve(y_test, probs)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("F1: ", metrics.f1_score(y_test, y_pred))
    print("ROC AUC Score:", metrics.roc_auc_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

    print(f'---------- {model_name} AFTER SMOTE----------')
    model.fit(X_smote, y_smote)
    y_pred = model.predict(X_test)

    probs_smoted = model.predict_proba(X_test)
    probs_smoted = probs_smoted[:, 1]
    fpr_smoted, tpr_smoted, _ = roc_curve(y_test, probs_smoted)
    precision_smoted, recall_smoted, _ = precision_recall_curve(y_test, probs_smoted)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("F1: ", metrics.f1_score(y_test, y_pred))
    print("ROC AUC Score:", metrics.roc_auc_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

    plt.plot(fpr_unsmoted, tpr_unsmoted, color='blue', label='UNSMOTEd')
    plt.plot(fpr_smoted, tpr_smoted, color='red', label='SMOTEd')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {model_name}')
    plt.legend()
    plt.show()

    plt.plot(recall_unsmoted, precision_unsmoted, color='blue', label='UNSMOTEd')
    plt.plot(recall_smoted, precision_smoted, color='red', label='SMOTEd')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curves - {model_name}')
    plt.legend()
    plt.show()
