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

#convert amount in euros
aud = 0.626093;
nzd = 0.591501;
sek = 0.0935468;
gbp = 1.16536;
mxn = 0.0467946;
conversion_rate = {"AUD": 0.626093,
                    "NZD": 0.591501,
                    "SEK": 0.0935468,
                    "GBP": 1.16536,
                    "MXN": 0.0467946}

df['amount'] = df[['amount', 'currencycode']].apply(lambda row: row['amount']*conversion_rate[row['currencycode']], axis=1)


data_fraud = df.loc[df['simple_journal'] == 'Chargeback']
data_no_fraud = df.loc[df['simple_journal'] == 'Settled']
data_refused = df.loc[df['simple_journal'] == 'Refused']

ax = sns.distplot(np.array(data_fraud['amount']), hist = False, label = 'Chargeback', color = 'red')
ax = sns.distplot(np.array(data_refused['amount']), hist = False, label = 'Refused', color = 'lightgray')
ax = sns.distplot(np.array(data_no_fraud['amount']), hist = False, label = 'Settled', color = 'lightgreen')


ax.set(xlabel='Transaction amount')

target_img = r"../images/density_estimation_amount.pdf"
plt.savefig(target_img, format='pdf')
plt.show()
print(f"Image saved in 'images' folder.")
