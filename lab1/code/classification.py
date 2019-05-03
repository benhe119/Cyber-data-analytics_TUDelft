from imblearn.over_sampling import SMOTE
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics

DATA_PATH = r"../data/"
DATA_FILE = "data_processed.csv"

df = pd.read_csv(DATA_PATH + DATA_FILE)

#Let's split into features and labels
y = df['label']
X = df.drop(columns = ['label'])

#We first split in train and test set -> we only have to apply SMOTE on train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


#Apply SMOTE
print(f"Fraud cases: {len(y_train[y_train==1])} and non-fraud cases: {len(y_train[y_train==0])}")

smt = SMOTE()
X_smote, y_smote = smt.fit_sample(X_train, y_train)

print(f"Fraud cases: {len(y_smote[y_smote == 1])} and non-frau cases: {len(y_smote[y_smote ==0 ])}")


#Try some classification
print('---------- CLASSIFICATION BEFORE SMOTE----------')

print('\n Random Forest')
rf=RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)

print("Accuracy RF:", metrics.accuracy_score(y_test, y_pred_rf))
print("Recall RF:", metrics.recall_score(y_test, y_pred_rf))
print("ROC AUC Score RF:",metrics.roc_auc_score(y_test, y_pred_rf))
print(metrics.confusion_matrix(y_test, y_pred_rf))


print('\n Logistic Regression')
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Accuracy log reg:", metrics.accuracy_score(y_test, y_pred_lr))
print("Recall log reg:", metrics.recall_score(y_test, y_pred_lr))
print("ROC AUC Score log reg:",metrics.roc_auc_score(y_test, y_pred_lr))
print(metrics.confusion_matrix(y_test, y_pred_lr))


print('\n Support Vector Machine')
svm = svm.SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("Accuracy SVM:", metrics.accuracy_score(y_test, y_pred_svm))
print("Recall SVM:", metrics.recall_score(y_test, y_pred_svm))
print("ROC AUC Score SVM:",metrics.roc_auc_score(y_test, y_pred_svm))
print(metrics.confusion_matrix(y_test, y_pred_svm))


print('---------- CLASSIFICATION AFTER SMOTE----------')

print('\n Random Forest')
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_smote,y_smote)
y_pred=clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy RF:",metrics.accuracy_score(y_test, y_pred))
print("Recall RF:", metrics.recall_score(y_test, y_pred))
print("ROC AUC Score RF:",metrics.roc_auc_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

print('\n Logistic Regression')
lr = LogisticRegression()
lr.fit(X_smote, y_smote)
y_pred_lr = lr.predict(X_test)

print("Accuracy log reg:", metrics.accuracy_score(y_test, y_pred_lr))
print("Recall log reg:", metrics.recall_score(y_test, y_pred_lr))
print("ROC AUC Score log reg:",metrics.roc_auc_score(y_test, y_pred_lr))
print(metrics.confusion_matrix(y_test, y_pred_lr))


print('\n Support Vector Machine')
svm = svm.SVC()
svm.fit(X_smote, y_smote)
y_pred_svm = svm.predict(X_test)
print("Accuracy SVM:", metrics.accuracy_score(y_test, y_pred_svm))
print("Recall SVM:", metrics.recall_score(y_test, y_pred_svm))
print("ROC AUC Score SVM:",metrics.roc_auc_score(y_test, y_pred_svm))
print(metrics.confusion_matrix(y_test, y_pred_svm))
