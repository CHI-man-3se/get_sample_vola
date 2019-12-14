import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from itertools import chain


# たまには絶対パス指定
df = pd.read_csv('/Users/apple/python/oanda/input_USDJPY_CSV/USDJPY_10m_after2014.csv')
df_len = len(df)
df.head()

##############################################################
###            ALLデータかランダムで100サンプル数だけ取得する      ###
##############################################################

def get_rand_sample_fromALLrate(df , sample_size):
    
    df_len = len(df)
    rand_num = random.randint(0, df_len-sample_size)

    date = df.iat[rand_num, 2]

    sample_set = df.iloc[rand_num:rand_num+sample_size,:]  
    
    return sample_set ,date


##############################################################
###       ALLデータから順々に100づつデータを持ってくる           ###
##############################################################

def get_full_sample_fromALLrate(df , sample_size):
    
    df_len = len(df)
    cnt_loop = df_len // sample_size

    sample_block = []
    
    for i in range(0, cnt_loop):
        startpoint = i*sample_size
        sample_set = df.iloc[1+startpoint:startpoint + sample_size+1, :]  
        sample_block.append(sample_set)
    
    return sample_block


##############################################################
###           やたら分散が大きいやつを取得する関数              ###
###    ここで、取得されるcandleはすでに大きいdiffがたされたあと    ###
##############################################################

def get_Theshold(sample_set):

    candle_std = sample_set['OPEN'].std()
    candle_mean = sample_set['OPEN'].mean()

    std_90line = 1.96 * candle_std
    under_Threshold = candle_mean - std_90line
    over_Threshold = candle_mean + std_90line

    high_volatility = sample_set.query('OPEN < @under_Threshold or @over_Threshold < OPEN')

    return under_Threshold , over_Threshold , high_volatility


##############################################################
###          分散が大きいやつのあとに来るcandleの属性を確認       ###
##############################################################

def get_highVolatility_index(sample_set):


    #print(sample_high_volatility)
    sample_len = len(sample_set)
    
    index_num = sample_set.index.values

    # 分散がでかい、先頭のindexは入れておく
    #drop_sequence_index = np.array([index_num[0]])
    drop_sequence_index = [index_num[0]]
    
    # 分散がでかくなった瞬間を持ってきたいので、連続indexが連続になっているのは省く
    for i in range(sample_len):

        # listがオーバフローしないために
        if i == sample_len-1:
            None
        else :
            #print("i  " , index_num[i])
            #print("i+1" , index_num[i+1])

            is_sequence = index_num[i+1] - index_num[i]

            # 連続しているindexは省き、分散がでかくなった瞬間のindexを持ってくる
            if is_sequence == 1:
                None
            else:
                #np.append(drop_sequence_index,index_num[i+1])
                drop_sequence_index.append(index_num[i+1])

    return drop_sequence_index



##############################################################
###           1階差をとり、nparrayに変換する関数                ###
##############################################################

# 引数　pandas
# 返り値　nparray
def get_diff_1(sample):
    open_array = sample['OPEN'].values
    diff_1_list = []
    j = 0
    for i in open_array:
        if j == 0:
            None
        else:
            diff_1_list.append(i - j)
        
        j = i

    diff_1_nparray = np.array(diff_1_list)

    return diff_1_nparray


##############################################################
###           　　　　　　　　連を確認する関数                ###
##############################################################

def verify_for_seaqential(diff_1_nparray):
    
    targetpoint = -1

    while targetpoint < 0:
        rand_num = random.randint(0, len(diff_1_nparray)-5)
        targetpoint = diff_1_nparray[rand_num]
      
    for i in range(1,6):
        
        if diff_1_nparray[rand_num+i] > 0:
            print("OK")
        else :
            print("NG") 




##############################################################
###             　分散の検出が、じわじわか、stayか      ###
##############################################################
def is_intense_before_highVola(highVola_rate_list):
    diff_rate = []

    UNDER = -0.06
    OVER = 0.06
    is_intense_before = 0
    is_bounce_after = 0

    list_is_intense_before = []
    list_is_bounce_after = []

    shortloopcnt = 0
    
    for i in highVola_rate_list:
        temp = np.array(i)
        diff_rate = np.diff(temp)
        #is_intense_array  = np.where( (diff_rate < UNDER) | (OVER < diff_rate))
        

        # diff が 0.06より大きいか小さいかで、 激しいのかどうかジャッジ
        # loopカウントが 3未満のときは、highVola検出の前　つまり、急にhighVolaになったのかじわじわなのか見極める
        # loopカウントが 4以上のときは、highVola検出の後　つまりhighVolaのあとにstayか、bounceか見極める
        for val in diff_rate:
            if shortloopcnt <= 3:
                if val < UNDER :
                    is_intense_before = 1
                elif val > OVER :
                    is_intense_before = 2
                else :
                    is_intense_before = 0
            else :
                if val < UNDER :
                    is_bounce_after = 1
                elif val > OVER :
                    is_bounce_after = 2
                else :
                    is_bounce_after = 0

            shortloopcnt = shortloopcnt + 1    
        
        list_is_intense_before.append(is_intense_before)
        list_is_bounce_after.append(is_bounce_after)
        shortloopcnt = 0
        
        print("break")
        
    return list_is_intense_before , list_is_bounce_after

##############################################################
###             　分散が大きくなったあと、bounce　か　stay      ###
##############################################################
def is_bounce_after_highVola():
    print(" temp ")


##############################################################
###             　分散がcandleの前後を撮ってくる               ###
###        サブのメソッドでじわじわ分散が大きくなったのか確認     ###
###        サブのメソッドで分散が大きくなったあとstayがbounceか確認     ###
##############################################################


def classify_volatility(highVolasample,df):

    ALL_highVola_index_diff_list = []
    EACH_highVola_index_diff_list = []
    
    ''' pandasで訓練データセットを作りたい '''
    ''' highVolasampleのindex '''
    ''' open価格 '''
    ''' 正解は、is_intense_before 、is_bounce_afterの組み合わせ '''
    
    ''' diffを持ってこないと '''
    
    # highVolaリストを回す
    for i in highVola_index_list:
        # highVolaのアラウンド 4を拾ってきてそれを新たなリストにする
        for j in range(-4,5):
            EACH_highVola_index_diff_list.append( df.iat[i+j,4])
        
        # 新たなリストを作ったら[ [-4:4], [-4:4], ... , [-4,4] ]
        ALL_highVola_index_diff_list.append(EACH_highVola_index_diff_list)
        
        EACH_highVola_index_diff_list = [] # ラウンド用のtempをクリアする

    list_is_intense_before, list_is_bounce_after = is_intense_before_highVola(ALL_highVola_index_diff_list)
    
    print(ALL_highVola_index_diff_list)
    print(list_is_intense_before)
    print(list_is_bounce_after)
 


###################################################################################
###################################################################################
###           　　　　　　　　            main文                                   ###
###################################################################################
###################################################################################

"""
SAMPLE_SIZE = 100

sample_set , sample_date = get_rand_sample_fromALLrate(df,SAMPLE_SIZE)

under_Threshold , over_Threshold , sample_high_volatility = get_Theshold(sample_set)

len_high_volatility = len(sample_high_volatility)

if len_high_volatility == 0:
    print("NO HIGH VOLATILITY about thins samples")
else :
    index_highVola = get_highVolatility_index(sample_high_volatility)
    print(index_highVola)
"""


##############################################################
###                   Fullのサンプルブロックを作る             ###
##############################################################

sample_block = get_full_sample_fromALLrate(df,100)
for i in sample_block:
    under_Threshold , over_Threshold , sample_high_volatility = get_Theshold(i)
    len_high_volatility = len(sample_high_volatility)

    if len_high_volatility == 0:
        print("NO HIGH VOLATILITY about thins samples")
    else :
        index_highVola = get_highVolatility_index(sample_high_volatility)
        print(index_highVola)
        
print(under_Threshold, over_Threshold)
print("debug")





##############################################################
###          分散が大きいindexを1000持ってくる                 ###
##############################################################

def get_index_highVola_forTrain():
    temp_train_index_highVola = []
    SAMPLE_SIZE = 100
    is_while = 0
    HIGHVOLASIZE = 100

    while(is_while < HIGHVOLASIZE):

        sample_set , sample_date = get_rand_sample_fromALLrate(df,SAMPLE_SIZE)
        under_Threshold , over_Threshold , sample_high_volatility = get_Theshold(sample_set)
        len_high_volatility = len(sample_high_volatility)
        
        if len_high_volatility == 0:
            #print("NO HIGH VOLATILITY about thins samples")
            None
        else :
            index_highVola = get_highVolatility_index(sample_high_volatility)
        
        temp_train_index_highVola.append(index_highVola)    # while loop判定用

        
        is_while = len(list(chain.from_iterable(temp_train_index_highVola)))

    train_index_highVola = list(chain.from_iterable(temp_train_index_highVola))
    sorted_list = list(set(train_index_highVola))

    return sorted_list

highVola_index_list = get_index_highVola_forTrain()

classify_volatility(highVola_index_list, df)