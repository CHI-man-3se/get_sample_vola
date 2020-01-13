import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rand


# たまには絶対パス指定
#df = pd.read_csv('/Users/apple/python/oanda/output_classified_csv/classified_sample_100.csv' ,index_col=0)
#df = pd.read_csv('/Users/apple/python/oanda/output_classified_csv/classified_sample_100_ALL.csv' ,index_col=0)
#df = pd.read_csv('/Users/apple/python/oanda/output_classified_csv/classified_sample_100_ALL_OPEN_CLOSE_THESHOLD.csv' ,index_col=0)

### ABSOLUTE ###
#df = pd.read_csv('/Users/apple/python/oanda/output_classified_csv/test_variation/classified_sample_100_ABSOLUTE_DIFF_THESHOLD_std1.csv' ,index_col=0)
#df = pd.read_csv('/Users/apple/python/oanda/output_classified_csv/test_variation/classified_sample_100_ABSOLUTE_DIFF_THESHOLD_std1_96.csv' ,index_col=0)
#df = pd.read_csv('/Users/apple/python/oanda/output_classified_csv/test_variation/classified_sample_100_ABSOLUTE_DIFF_THESHOLD_std1_64.csv' ,index_col=0)
#df = pd.read_csv('/Users/apple/python/oanda/output_classified_csv/test_variation/classified_sample_100_ABSOLUTE_DIFF_THESHOLD_std0_50.csv' ,index_col=0)


### RELATIVE ###
#df = pd.read_csv('/Users/apple/python/oanda/output_classified_csv/test_variation/classified_sample_RATE_THRESHOLD1_64_RELATIVE_DIFF_THESHOLD_1_0.csv' ,index_col=0)
df = pd.read_csv('/Users/apple/python/oanda/output_classified_csv/test_variation/classified_sample_RATE_THRESHOLD_NONE_RELATIVE_DIFF_THESHOLD_1_0.csv' ,index_col=0)


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

    sum_hitlist = sum(hit_list)
    print(hit_list,sum_hitlist)
    
    plt.bar(catego_list, hit_list)
    plt.show()
   
##########################################################
###         beforeのクラスを指定しafterに傾向があるか確認    ###
##########################################################
def getHitNum_diffClass_from_Df(df, catego1):

    catego_list = ['U_diff','D_diff','F_diff','U_little_diff','D_little_diff']

    hit_list = [0,0,0,0,0]

    for i in range( len(df) ):    
        if (df.loc[i,'classdiff_before'] == catego1) and (df.loc[i,'classdiff_after'] == catego_list[0]):
            hit_list[0] = hit_list[0] + 1
        elif (df.loc[i,'classdiff_before'] == catego1) and (df.loc[i,'classdiff_after'] == catego_list[1]):
            hit_list[1] = hit_list[1] + 1
        elif (df.loc[i,'classdiff_before'] == catego1) and (df.loc[i,'classdiff_after'] == catego_list[2]):
            hit_list[2] = hit_list[2] + 1
        elif (df.loc[i,'classdiff_before'] == catego1) and (df.loc[i,'classdiff_after'] == catego_list[3]):
            hit_list[3] = hit_list[3] + 1
        elif (df.loc[i,'classdiff_before'] == catego1) and (df.loc[i,'classdiff_after'] == catego_list[4]):
            hit_list[4] = hit_list[4] + 1

        else:
            None
    sum_hitlist = sum(hit_list)
    
    
    return hit_list, sum_hitlist

##########################################################
###                    グラフをプロット　　　　　　　　　　　###
##########################################################

def discribe_category_data(hit_list, diff_catego_list, hitsum_list_each):
    #plt.bar(catego_list, hit_list)
    #plt.show()
    
    ## y軸の最大範囲を先に指定する
    adopt_MAX = 0
    for tmp in hit_list:
        for i in tmp:
            if adopt_MAX < i:
                adopt_MAX = i
            else:
                None

    cnt = 0

    for i,j in zip(diff_catego_list,hit_list):
        cnt = cnt + 1
        plt.rcParams["font.size"] = 8
        plt.subplot(2,3,cnt)
        plt.title(i)
        plt.bar(diff_catego_list, j)
        plt.ylim(0,adopt_MAX)


    plt.show()


##########################################################
###          傾向が時間依存していないか、randomに確認する    ###
##########################################################

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

catego_list = ['U', 'D', 'F', 'V']
diff_catego_list = ['U_diff','D_diff','F_diff','U_little_diff','D_little_diff']

hit_list = []
hitsum_list_each = []

print(len(df))
print(df.head())
print("for copy about columns name")
print(df.columns)

print("main routine start")
print("\n")

## columnsのクラス名がいくつあるのか　足し算
"""
category_Before_num = get_category_number_from_DF(df, "classdiff_before")
category_After_num = get_category_number_from_DF(df, "classdiff_after")
print(category_Before_num)
print(category_After_num)
"""

## 引数にbeforeのラベル名を指定しafterがどうなるか棒グラフで視覚化
## カテゴリーに対して
"""
get_categoryFeature_from_Df(df,"D")
get_categoryFeature_from_Df(df,"U")
get_categoryFeature_from_Df(df,"V")
get_categoryFeature_from_Df(df,"F")
"""

## 引数にbeforeのラベル名を指定しafterがどうなるか棒グラフで視覚化
## 実際のdiffに対して
"""
hit_list_D_diff, sum_len_D_diff = getHitNum_diffClass_from_Df(df,"D_diff")
hit_list_U_diff, sum_len_U_diff = getHitNum_diffClass_from_Df(df,"U_diff")
hit_list_F_diff, sum_len_F_diff = getHitNum_diffClass_from_Df(df,"F_diff")
hit_list_Dl_diff, sum_len_D_ldiff = getHitNum_diffClass_from_Df(df,"D_little_diff")
hit_list_Ul_diff, sum_len_Ul_diff = getHitNum_diffClass_from_Df(df,"U_little_diff")
"""

for i in diff_catego_list:
    tmp_hit_list, tmphitsum = getHitNum_diffClass_from_Df(df,i)

    hit_list.append(tmp_hit_list)
    hitsum_list_each.append(tmphitsum)

print(hit_list)
print(hitsum_list_each)

discribe_category_data(hit_list, diff_catego_list, hitsum_list_each)

"""
for i in range(10):
    Rand_get_categoryFeature_from_Df(df,"F")
"""