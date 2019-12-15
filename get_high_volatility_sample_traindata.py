import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from itertools import chain


# たまには絶対パス指定
df = pd.read_csv('/Users/apple/python/oanda/input_USDJPY_CSV/USDJPY_10m_after2014.csv')
df_len = len(df)
df.head()
print("break")


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

    return under_Threshold , over_Threshold



##############################################################
###           分散が大きいindex番号だけをとってくる              ###
##############################################################

def get_highVolatility_index(sample_set ,under_Threshold, over_Threshold):

    # high_volaのdataframeをゲット　※ほしいのはindexだけなので、それだけ貰えればいいかも
    high_volatility = sample_set.query('OPEN < @under_Threshold or @over_Threshold < OPEN')

    sample_len = len(high_volatility)
    
    index_num = high_volatility.index.values

    drop_sequence_index = []

    # 分散がでかくなった瞬間を持ってきたいので、連続indexが連続になっているのは省く
    # indexが連続しているのをdropさせるloop

    # 分散がでかい、先頭のindexは無条件で追加
    #drop_sequence_index = [index_num[0]]
    
    if sample_len == 0 :
        None
    else:
        drop_sequence_index = [index_num[0]]

        for i in range(sample_len):

            # listがオーバフローしないために
            if i == sample_len-1:
                None
            else :
                
                is_sequence = index_num[i+1] - index_num[i]
                # 連続しているindexは省き、分散がでかくなった瞬間のindexを持ってくる
                if is_sequence == 1:
                    None
                else:
                    drop_sequence_index.append(index_num[i+1])

    return drop_sequence_index



##############################################################
###      分散が大きいindex番号から、around list BLOCK作成       ###
##############################################################

def get_high_vola_Blocks(df , high_vola_index):
    
    df_high_vola_Blocks = []

    for i in high_vola_index:
        df_high_vola = df.iloc[ i-6:i+6, :]  
        df_high_vola_Blocks.append(df_high_vola)
        
    return df_high_vola_Blocks

##############################################################
###      　　　　　　　　　index と OPENだけにする　　　　       ###
##############################################################

def drop_other(highVola_Blocks):

    print(highVola_Blocks[0].columns)
    print("break")

    listof_df_droped = []

    for i in highVola_Blocks:
        print(i.columns)
        print(i)
    
    #return listof_df_droped
    
    
##############################################################
###           1階差をとり、nparrayに変換する関数                ###
##############################################################

# 引数　pandas
# 返り値　nparray
def get_diff_1(sample):
    print(sample)
    print(type(sample[0]))
    print("debug")
    open_array = sample[0].iloc[:,4].values
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
###                nparrayの階差関数をつかって                ###
###     inputのblocksがpandasを要素とするlistなので            ###
###     　　　　　for 文をつかって要素に分解してあげる必要がある   ###
##############################################################

def get_diff1_from_Blocks(blocks):

    diff_each_blocks = []

    for i in blocks:
        np_blocks = i.iloc[:,4].values
        diff_each_blocks.append( np.diff(np_blocks) )

    return diff_each_blocks


############################################################
###          diffを見て、それがintense or 緩やかを確認したい   ###
###     sample 100ごとの diff　listを作ってそれのstdをとるか   ###
#############################################################

def get_ALLdiff(sample_set):

    np_sample = sample_set.iloc[:,4].values
    np_diff_sample = np.diff(np_sample)
    X = np.array(range(99))
    
    diff_std = np_diff_sample.std()
    diff_mean = np_diff_sample.mean()

    point_1sigma_over = diff_mean + diff_std
    point_1sigma_under = diff_mean - diff_std
    point_2sigma_over = diff_mean + 2*diff_std
    point_2sigma_under = diff_mean - 2*diff_std

    ''' diffを棒グラフで表示する
    plt.bar(X, np_diff_sample)
    plt.show()
    '''

    return point_2sigma_under , point_2sigma_over


############################################################
###          diffを見て、それがintense or 緩やかを確認したい   ###
###     sample 100ごとの diff　listを作ってそれのstdをとるか   ###
#############################################################

def is_intense_diff(point_2sigma_under, point_2sigma_over, diff_blocks):

    np_diff_blocks = np.array(diff_blocks)
    
    print(type(diff_blocks))
    print(type(np_diff_blocks))
    print(len(np_diff_blocks))

    print(point_2sigma_under)
    print(point_2sigma_over)

    print(diff_blocks)
    
    judged_array_under = np.where((point_2sigma_under<np_diff_blocks) , False , True)
    judged_array_over = np.where((np_diff_blocks<point_2sigma_over) , False , True)

    print(judged_array_under)
    print(judged_array_over)
    """
    for i in np_diff_blocks:
       # target_index = diff_blocks[5] ## 検出した瞬間のindex　前後での、isintenseを検出したいので大事

        print(len(i))

        for i in range( len(i) ): ## マクロで11を直接指定しても良き

    """
    print("break")






##############################################################
###           　　　　　　　　連を確認する関数 　               ###
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

sample_blocks = get_full_sample_fromALLrate(df,100)

# 100ごとのsampleブロックから、しきい値を検出し、high_Vola のaroudをgetする
for i in sample_blocks:

    under_Threshold , over_Threshold = get_Theshold(i)
    
    
    highVola_index = get_highVolatility_index(i , under_Threshold , over_Threshold)




    # 分散が大きい大きいindexが存在しなかったらパスする
    if len(highVola_index) == 0:
        None
    else:
        
        highVola_Blocks = get_high_vola_Blocks(df, highVola_index)
            
        diff_blocks = get_diff1_from_Blocks(highVola_Blocks)
        
        point_2sigma_under, point_2sigma_over = get_ALLdiff(i)

        is_intense_diff(point_2sigma_under, point_2sigma_over, diff_blocks)
    

    



