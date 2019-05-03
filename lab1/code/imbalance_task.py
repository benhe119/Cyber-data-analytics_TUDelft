from imblearn.over_sampling import SMOTE
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

DATA_PATH = r"../data/"
DATA_FILE = "data_for_student_case.csv"
DATA_ZIP = "data_for_student_case.csv.zip"

exists = os.path.isfile(DATA_PATH + DATA_FILE)
if not exists:
    print("Data file not existing, unzipping file...")
    zip_ref = zipfile.ZipFile(DATA_PATH + DATA_ZIP, 'r')
    zip_ref.extractall(DATA_PATH)
    zip_ref.close()
    print("Done.")
else:
    print("Data file found.")

df = pd.read_csv(DATA_PATH + DATA_FILE)

data_train = df.loc[(df['simple_journal'] == 'Settled') | (df['simple_journal'] == 'Chargeback')].copy()
data_train.loc[data_train['simple_journal'] == 'Settled']['simple_journal'] = 0
data_train.loc[data_train['simple_journal'] == 'Chargeback']['simple_journal'] = 1

#print(data_train.groupby('simple_journal').count())

#Let's split into features and labels
y = data_train['simple_journal']
x = data_train.drop('simple_journal', axis=1)

print(f"Number of records of fraud {y.value_counts()[1]} and non-fraud {y.value_counts()[0]} before SMOTE.")

print("Apply SMOTE")
smt = SMOTE()
x_smote, y_smote = smt.fit_sample(x, y)

print(f"Number of records of fraud {np.bincount(y_smote)[1]} and non-fraud {np.bincount(y_smote)[0]} after SMOTE.")


#Try some classification
print('---------- CLASSIFICATION BEFORE SMOTE----------')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("ROC AUC Score:",metrics.roc_auc_score(y_test, y_pred))



print('---------- CLASSIFICATION AFTER SMOTE----------')

X_train, X_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size=0.3)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("ROC AUC Score:",metrics.roc_auc_score(y_test, y_pred))


