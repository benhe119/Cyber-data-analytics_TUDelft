# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:51:59 2019

@author: gabri
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_corr(df, file_name):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=70, fontsize=8);
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8);
    plt.savefig(r"../images/correlation_matrix_{}.pdf".format(file_name))
    plt.show()
    
def plot_corr_sns(df, file_name):
    sns.set()
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(
            corr, 
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True,
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        fontsize=10
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontsize=10
    )
    plt.savefig(r"../images/correlation_matrix_{}.pdf".format(file_name))
    plt.show()
    
    
file_name = 'test_data'
DATA_PATH = r"C:../data/{}.csv".format(file_name)
data = pd.read_csv(DATA_PATH)


data_no_discr = data.drop(columns=([f'S_PU{num+1}' for num in range(11)] + ['S_V2']))
plot_corr_sns(data_no_discr, file_name)

