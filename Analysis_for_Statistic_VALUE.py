import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rand


# たまには絶対パス指定
df = pd.read_csv('/Users/apple/python/oanda/output_classified_csv/Statistic_VALUE_fromALL.csv' ,index_col=0)  ## index_colを0にしておかないと、更にindexがふえてしまう
#df = pd.read_csv('/Users/apple/python/oanda/output_classified_csv/Statistic_VALUE_fromALL.csv')


def discribe_graph(np_list):
    X = np.arange(0,len(np_list))
    
    plt.plot(X, np_list)
    plt.show()

def discribe_hist(np_list):
    plt.hist(np_list, bins=200)
    plt.show()

###################################################################################
###################################################################################
###           　　　　　　　　            main文                                   ###
###################################################################################
###################################################################################


print(len(df))
print(df.head())
print("for copy about columns name")
print(df.columns)

print("main routine start")
print("\n")

print(df.describe())

"""
discribe_graph( df.loc[:,'candle_mean'].values )
discribe_graph( df.loc[:,'candle_std'].values )
discribe_graph( df.loc[:,'diff_mean'].values )
discribe_graph( df.loc[:,'diff_std'].values )
"""

discribe_hist( df.loc[:,'candle_mean'].values )
discribe_hist( df.loc[:,'candle_std'].values )
discribe_hist( df.loc[:,'diff_mean'].values )
discribe_hist( df.loc[:,'diff_std'].values )