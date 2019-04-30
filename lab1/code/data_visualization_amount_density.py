# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 09:51:53 2019

@author: Bianca Iancu
"""


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile

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

data_fraud = df.loc[df['simple_journal'] == 'Chargeback']
data_no_fraud = df.loc[df['simple_journal'] == 'Settled']
data_refused = df.loc[df['simple_journal'] == 'Refused']

sns.set_style('whitegrid')
ax = sns.distplot(np.array(data_fraud['amount']), hist = False, label = 'Chargeback', color = 'red')

sns.set_style('whitegrid')
ax = sns.distplot(np.array(data_refused['amount']), hist = False, label = 'Refused', color = 'blue')

sns.set_style('whitegrid')
ax = sns.distplot(np.array(data_no_fraud['amount']), hist = False, label = 'Settled', color = 'green')



ax.set(xlabel='Transaction amount')

target_img = r"../images/density_estimation_amount.pdf"
plt.savefig(target_img, format = 'pdf')
print(f"Image saved in 'images' folder.")
