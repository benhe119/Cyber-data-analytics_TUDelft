# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:23:31 2019

@author: gabri
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = r".\data_for_student_case.csv"

data = pd.read_csv(DATA_PATH)


categories = data['txvariantcode'].unique()
num_of_categories = len(categories)
fraud = []
no_fraud = []
refused = []

for category in categories:
    print(f"Category {category}")
    cat_data = data[data['txvariantcode'] == category]
    fraud.append(cat_data[cat_data['simple_journal'] == 'Chargeback'].shape[0])
    no_fraud.append(cat_data[cat_data['simple_journal'] == 'Settled'].shape[0])
    refused.append(cat_data[cat_data['simple_journal'] == 'Refused'].shape[0])
    
ind = np.arange(num_of_categories)    # the x locations for the groups
width = 0.5      # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, refused, width, color='lightgray')
p2 = plt.bar(ind, fraud, width, bottom=np.array(refused), color='red')
p3 = plt.bar(ind, no_fraud, width, bottom= np.array(fraud) + np.array(refused), color='lightgreen')


plt.ylabel('Amount of transactions')
plt.title('Transactions per card type')
plt.xticks(ind, tuple(categories), fontsize=5)
plt.yscale('log')
plt.xticks(rotation=30)
plt.legend((p1[0], p2[0], p3[0]), ('Refused', 'Chargeback', 'Settled'))
plt.savefig('stacked_bar.pdf', format='pdf')
plt.show()
