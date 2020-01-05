import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rand


# たまには絶対パス指定
#df = pd.read_csv('/Users/apple/python/oanda/output_classified_csv/classified_sample_100.csv' ,index_col=0)
df = pd.read_csv('/Users/apple/python/oanda/output_classified_csv/classified_sample_100_ALL.csv' ,index_col=0)


""""""""""""""""""""
'データを解析する関数'
""""""""""""""""""""


def get_category_number_from_DF(df, columns_name):

    Se_hit = df[columns_name].value_counts()
    
    return Se_hit


def get_categoryFeature_from_Df(df, catego1):

    catego_list = ['U', 'D', 'F', 'V']

    hit_list = [0,0,0,0]

    for i in range( len(df) ):    
        if (df.iloc[i,0] == catego1) and (df.iloc[i,1] == catego_list[0]):
            hit_list[0] = hit_list[0] + 1
        elif (df.iloc[i,0] == catego1) and (df.iloc[i,1] == catego_list[1]):
            hit_list[1] = hit_list[1] + 1
        elif (df.iloc[i,0] == catego1) and (df.iloc[i,1] == catego_list[2]):
            hit_list[2] = hit_list[2] + 1
        elif (df.iloc[i,0] == catego1) and (df.iloc[i,1] == catego_list[3]):
            hit_list[3] = hit_list[3] + 1
        else:
            None
    print(hit_list)
    
    plt.bar(catego_list, hit_list)
    plt.show()
   


def Rand_get_categoryFeature_from_Df(df, catego1):

    catego_list = ['U', 'D', 'F', 'V']

    hit_list = [0,0,0,0]

    hit_rate = [0,0,0,0]

    #start = rand.randint(0, len(df)-500 )
    start = rand.randint(3376, 6387 )
    
    #end = start + 500
    end = start + 300
    sum_hit_catego = 0

    for i in range( start, end ):    
        if (df.iloc[i,0] == catego1) and (df.iloc[i,1] == catego_list[0]):
            hit_list[0] = hit_list[0] + 1
            sum_hit_catego = sum_hit_catego + 1
        elif (df.iloc[i,0] == catego1) and (df.iloc[i,1] == catego_list[1]):
            hit_list[1] = hit_list[1] + 1
            sum_hit_catego = sum_hit_catego + 1
        elif (df.iloc[i,0] == catego1) and (df.iloc[i,1] == catego_list[2]):
            hit_list[2] = hit_list[2] + 1
            sum_hit_catego = sum_hit_catego + 1
        elif (df.iloc[i,0] == catego1) and (df.iloc[i,1] == catego_list[3]):
            hit_list[3] = hit_list[3] + 1
            sum_hit_catego = sum_hit_catego + 1
        else:
            None

    for i in range( len(hit_list) ):
        hit_rate[i] = hit_list[i]/sum_hit_catego
        
    hit_rate = np.round(hit_rate, 4)

    print("startINDEX" , start)
    print("list--->" , hit_list)
    print("rate--->" , hit_rate)

    '''    
    plt.bar(catego_list, hit_list)
    plt.show()
    '''

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

category_Before_num = get_category_number_from_DF(df, "category_Before")
category_After_num = get_category_number_from_DF(df, "category_After")

print(category_Before_num)
print("\n")
print(category_After_num)

debug_sum = 0

"""
get_categoryFeature_from_Df(df,"D")
get_categoryFeature_from_Df(df,"U")
get_categoryFeature_from_Df(df,"V")
get_categoryFeature_from_Df(df,"F")
"""

for i in range(10):
    Rand_get_categoryFeature_from_Df(df,"F")