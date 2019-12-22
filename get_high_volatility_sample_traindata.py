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

    return point_1sigma_under , point_1sigma_over





############################################################
###          numpyを使わず普通のリストで行う                  ###
###          diffを見て、それがintense or 緩やかを確認したい   ###
###     sample 100ごとの diff　listを作ってそれのstdをとるか   ###
#############################################################

def is_intense_diff(point_2sigma_under, point_2sigma_over, diff_blocks):

    is_under_before = 0
    is_under_after = 0
    is_over_before = 0
    is_over_after = 0

    classified_each_blocks = []
    # FF -> ~~
    # FU -> ~↗
    # FD -> ~↘
    # UF -> ↗~
    # UU -> ↗↗
    # UD -> ↗↘
    # DF -> ↘~
    # DU -> ↘↗
    # DD -> ↘↘
    # EE -> [-6 0] or [0 6]の短い間にupdownのいづれも経験すること　多分そんなことはないと思うが。。。

    under_indexs_each = []
    over_indexs_each = []
    under_indexs_SAMPLE = []
    over_indexs_SAMPLE = []

    print("***  start  ***")
    print(len(diff_blocks))

    print(point_2sigma_under)
    print(point_2sigma_over)

    ## [-6 6]のdiff blockが複数個あるのでまずはそれを取り出すfor文
    for each_block in diff_blocks:

        print(each_block)

        #[-6 6]の中を回すfor文
        for i in range(len(each_block)):

            #[-6 6]の中から、under Thresholdを検出しに行く
            if each_block[i] < point_2sigma_under:
                under_indexs_each.append(i)
                if i<=5:
                    is_under_before = 1
                else:
                    is_under_after = 1
            else :
                None

            #[-6 6]の中から over Thresholdを検出しに行く
            if each_block[i] > point_2sigma_over:
                over_indexs_each.append(i)
                if i<=5:
                    is_over_before = 1
                else:
                    is_over_after = 1
            else :
                None

        # beforeがいづれもフラット
        if (is_under_before==0) and (is_over_before==0):
            
            if (is_under_after==0) and (is_over_after==0):
                classified_each_blocks.append('FF')
            elif (is_under_after==0) and (is_over_after==1):
                classified_each_blocks.append('FU')
            elif (is_under_after==1) and (is_over_after==0):
                classified_each_blocks.append('FD')
            else:
                classified_each_blocks.append("FV")
        
        # beforeがUPのとき
        elif (is_under_before==0) and (is_over_before == 1):
            
            if (is_under_after==0) and (is_over_after==0):
                classified_each_blocks.append('UF')
            elif (is_under_after==0) and (is_over_after==1):
                classified_each_blocks.append('UU')
            elif (is_under_after==1) and (is_over_after==0):
                classified_each_blocks.append('UD')
            else:
                classified_each_blocks.append("UV")
    
        # beforeがDOWNのとき
        elif (is_under_before==1) and (is_over_before == 0):
            
            if (is_under_after==0) and (is_over_after==0):
                classified_each_blocks.append('DF')
            elif (is_under_after==0) and (is_over_after==1):
                classified_each_blocks.append('DU')
            elif (is_under_after==1) and (is_over_after==0):
                classified_each_blocks.append('DD')
            else:
                classified_each_blocks.append("DV")
        
        else:
            if (is_under_after==0) and (is_over_after==0):
                classified_each_blocks.append('VF')
            elif (is_under_after==0) and (is_over_after==1):
                classified_each_blocks.append('VU')
            elif (is_under_after==1) and (is_over_after==0):
                classified_each_blocks.append('VD')
            else:
                classified_each_blocks.append("VV")

        under_indexs_SAMPLE.append(under_indexs_each)
        over_indexs_SAMPLE.append(over_indexs_each)

        # 1つのSAMPLE群の eachごとにリセットする
        under_indexs_each = []
        over_indexs_each = []

        is_under_before = 0
        is_under_after = 0
        is_over_before =0
        is_over_after =0

    '''
    print("under before" , is_under_before)
    print("over before" ,is_over_before)
    print("under after" ,is_under_after)
    print("over after" ,is_over_after)
    print(under_indexs_SAMPLE)
    print(over_indexs_SAMPLE)
    print(classified_each_blocks)
    '''

    return classified_each_blocks



##############################################################
###           　  　　　データを作るために加工する               ###
##############################################################

def get_pandasDF_for_train(diff_blocks, classified_each_blocks, DF_train):

    #print("~~~ PD ~~~")
    #print(classified_each_blocks)
    #print(type(classified_each_blocks))

    #print(DF_train)
    #print("break")

    for i in classified_each_blocks:

        tmp_se = pd.Series( i , index=DF_train.columns )
        DF_train = DF_train.append( tmp_se, ignore_index=True )

    
    return DF_train



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

DF_train = pd.DataFrame( columns=['category'] ) #このdataFrameに対して、for文の中でデータを追加していく

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

        classified_each_blocks = is_intense_diff(point_2sigma_under, point_2sigma_over, diff_blocks)
    
        # pandas DFをmain文で定義して、DFを引数としてpandas作成関数に渡す
        # こうすることによって、違うSAMPLE BLOCKSに対しても同じDFにデータを入れることができる
        
        DF_train = get_pandasDF_for_train(diff_blocks, classified_each_blocks, DF_train)

print(DF_train)


