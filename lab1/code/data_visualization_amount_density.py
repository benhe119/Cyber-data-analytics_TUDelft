# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 09:51:53 2019

@author: Bianca Iancu
"""


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"./data_for_student_case.csv")

data_fraud = df.loc[df['simple_journal'] == 'Chargeback']
data_no_fraud = df.loc[df['simple_journal'] == 'Settled']
data_refused = df.loc[df['simple_journal'] == 'Refused']

sns.set_style('whitegrid')
ax = sns.distplot(np.array(data_fraud['amount']), hist = False, label = 'Chargeback', color = 'red')

sns.set_style('whitegrid')
ax = sns.distplot(np.array(data_no_fraud['amount']), hist = False, label = 'Settled', color = 'lightgreen')

sns.set_style('whitegrid')
ax = sns.distplot(np.array(data_refused['amount']), hist = False, label = 'Refused', color = 'lightgrey')

ax.set(xlabel='Transaction amount')

plt.savefig(r"./images/density_estimation_amount.pdf", format = 'pdf')
