import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from itertools import chain
from scipy import stats


from tqdm import tqdm

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
        sample_set = df.iloc[startpoint:startpoint + sample_size, :]  
        sample_block.append(sample_set)
    
    return sample_block


##############################################################
###           やたら分散が大きいやつを取得する関数              ###
###    ここで、取得されるcandleはすでに大きいdiffがたされたあと    ###
##############################################################

def get_Theshold(sample_set):

    candle_std = sample_set['OPEN'].std()
    candle_mean = sample_set['OPEN'].mean()

    #std_90line = 1.96 * candle_std
    std_90line = 1.64 * candle_std
    under_Threshold = candle_mean - std_90line
    over_Threshold = candle_mean + std_90line

    true_under_Threshold = candle_mean - std_90line
    true_over_Threshold = candle_mean + std_90line
    
    #return under_Threshold , over_Threshold
    return true_under_Threshold , true_over_Threshold , candle_mean, candle_std 


##############################################################
###           分散が大きいindex番号だけをとってくる              ###
##############################################################

def get_highVolatility_index(sample_set ,under_Threshold, over_Threshold):

    # high_volaのdataframeをゲット　※ほしいのはindexだけなので、それだけ貰えればいいかも
    high_volatility = sample_set.query('OPEN < @under_Threshold or @over_Threshold < OPEN')

    sample_len = len(high_volatility)
    
    index_num = high_volatility.index.values
    #open_rate = high_volatility.loc[:,'OPEN']

    open_rate = []
    drop_sequence_index = []
    extreme_point = []

    # 分散がでかくなった瞬間を持ってきたいので、連続indexが連続になっているのは省く
    # indexが連続しているのをdropさせるloop
    if sample_len == 0 :
        None
    else:
        
        if 0<=index_num[0]<=6 : ## high vola検出のindexが6以下のときはeach blocksをけいせいできないためここもパスする
            None
        else:
            drop_sequence_index = [index_num[0]] ## SAMPLE BLOCK100毎のhigh vola INDEXの先頭をreturn用のlistに追加する

        for i in range(sample_len):

            # listがオーバフローしないために
            if i == sample_len-1:
                None
            elif 0<=index_num[i]<=1 : ## high vola検出のindexが6以下のときはeach blocksをけいせいできないためここもパスする
                None
            else :  ##連続している、indexを省く
                is_sequence = index_num[i+1] - index_num[i]

                # 連続しているindexは省き、分散がでかくなった瞬間のindexを持ってくる
                if is_sequence < 6:
                    None
                else:
                    drop_sequence_index.append(index_num[i+1])


    for i in drop_sequence_index:
        rate = high_volatility.loc[i,'OPEN']
        open_rate.append(rate)

        if rate < under_Threshold:
            extreme_point.append('UNDER')
        elif rate > over_Threshold:
            extreme_point.append('OVER')
        else:
            extreme_point.append('UNEXPETED')

    return drop_sequence_index , open_rate , extreme_point



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
###           diffの連続性を確認するための関数                ###
###           4連続だったら、Seqフラグをセットする               ###
###           引数は、np array 　　　　　　　　                ###
###           返り値は、before afterの連続数　                ###
##############################################################

def get_Sequencial(diff_block):
    
    len_diff_block = len(diff_block)
    
    cnt_before_p = 0
    cnt_before_m = 0
    cnt_after_p = 0
    cnt_after_m = 0
    f_before_p = 0
    f_before_m = 0
    f_after_p = 0
    f_after_m = 0

    for i in range(len_diff_block):
        if i <= 5:
            ##### before #####
            if diff_block[i] >= 0:
                if f_before_p == 1:
                    cnt_before_p = cnt_before_p + 1
                else :
                    #cnt_before_p = 0
                    None
                f_before_p = 1
                f_before_m = 0

            else :
                if f_before_m == 1:
                    cnt_before_m = cnt_before_m + 1
                else :
                    #cnt_before_m = 0
                    None
                
                f_before_m = 1
                f_before_p = 0

        else:
            ##### after #####
            if diff_block[i] >= 0:
                if f_after_p == 1:
                    cnt_after_p = cnt_after_p + 1
                else :
                    cnt_after_p = 0
                
                f_after_p = 1
                f_after_m = 0

            else :
                if f_after_m == 1:
                    cnt_after_m = cnt_after_m + 1
                else :
                    cnt_after_m = 0
                
                f_after_m = 1
                f_after_p = 0

    return cnt_before_p, cnt_before_m, cnt_after_p, cnt_after_m
    
##############################################################
###           1階差をとり、nparrayに変換する関数                ###
##############################################################

# 引数　pandas
# 返り値　nparray
def get_diff_1_ALLrate(sample):
    
    #np_open_array = sample.iloc[:,2].values
    np_open_array = sample.loc[:,'OPEN'].values
    temp_np_open_diff = np.diff(np_open_array)
    np_open_diff = np.round(temp_np_open_diff, 3)
    np_open_diff = np.insert(np_open_diff, 0, 0)

    return np_open_diff


##############################################################
###                nparrayの階差関数をつかって                ###
###     inputのblocksがpandasを要素とするlistなので            ###
###     　　　　　for 文をつかって要素に分解してあげる必要がある   ###
##############################################################

def get_diff1_from_SAMPLE_Blocks(blocks , finalrate_in_sample):

    diff_each_blocks = []

    # numpy array  から listに変換する
    # diff をとった後に丸め誤差を考慮する必要がある
    # listの中に、numpy arrayがひとつ入っているのはだめなので、要素を分解してlistにする
    # listの先頭に0を追加しているのは、pandas DFに入れるときにどことのdiffとってるんやお前状態になるのを避けるため

    np_blocks = blocks.loc[:,'OPEN'].values
    
    ret_finalval = np_blocks[99]

    ## final_rateが0 つまり、1回目のSAMPLE BLOCKのときは　listに追加するのをやめる　diffが狂うので
    if final_rate == 0:
        None
    else:
        np_blocks = np.insert(np_blocks,0, finalrate_in_sample,axis=None)
   
    tmp_np_diff = np.diff(np_blocks)
    np_diff = np.round(tmp_np_diff, 3)
    diff_each_blocks = np_diff.tolist()
   
    return np_diff , ret_finalval

############################################################
###          diffを見て、それがintense or 緩やかを確認したい   ###
###     sample 100ごとの diff　listを作ってそれのstdをとるか   ###
#############################################################

def get_Limit_Each_Blocks(diff_sample , size_sigma):

    diff_std = diff_sample.std()
    diff_mean = diff_sample.mean()

    if size_sigma == 1: 
        point_sigma_over = diff_mean + diff_std
        point_sigma_under = diff_mean - diff_std
    elif size_sigma == 2:
        point_sigma_over = diff_mean + 2*diff_std
        point_sigma_under = diff_mean - 2*diff_std
    else:
        None

    return point_sigma_under , point_sigma_over , diff_mean, diff_std 

############################################################
###          numpyを使わず普通のリストで行う                  ###
###          diffを見て、それがintense or 緩やかを確認したい   ###
###     sample 100ごとの diff　listを作ってそれのstdをとるか   ###
#############################################################

def get_Is_Intense_diff(point_2sigma_under, point_2sigma_over, diff_blocks, diff_std):

    is_under_before = 0
    is_under_after = 0
    is_over_before = 0
    is_over_after = 0

    classified_each_blocks = []
    classified_each_blocks_before = []
    classified_each_blocks_after = []

    under_indexs_each = []      #diffブロック[-6,6]に対してすべて しきい値超えを判定する
    over_indexs_each = []       #diffブロック[-6,6]に対してすべて しきい値超えを判定する
    under_indexs_SAMPLE = []    #上のdiffブロック[-6,6]を1SAMPLEごとに持つ [[-6,6],[-6,6] , ,,]
    over_indexs_SAMPLE = []     #上のdiffブロック[-6,6]を1SAMPLEごとに持つ [[-6,6],[-6,6] , ,,]
    open_diff = []              # -6〜0のdiffを足したもののlist
    close_diff = []             # 0〜6のdiffを足したもののlist
    classified_diff_before = [] # diffによって真のup downを判定する
    classified_diff_after = []  # diffによって真のup downを判定する

    sum_diff_before = 0
    sum_diff_after = 0

    DIFF_LIMIT = (diff_std/2)

    ## [-6 6]のdiff blockが複数個あるのでまずはそれを取り出すfor文
    for each_block in diff_blocks:

        print(each_block)

        #[-6 6]の中を回すfor文
        for i in range(len(each_block)):

            #[-6 6]の中から、under Thresholdを検出しに行く
            if each_block[i] < point_2sigma_under:
                under_indexs_each.append(i)
                if i<=5: ## before
                    is_under_before = 1
                    sum_diff_before = sum_diff_before + each_block[i]
                else: ## after
                    is_under_after = 1
                    sum_diff_after = sum_diff_after + each_block[i]

            #[-6 6]の中から over Thresholdを検出しに行く
            elif each_block[i] > point_2sigma_over:
                over_indexs_each.append(i)
                if i<=5: ## before
                    is_over_before = 1
                    sum_diff_before = sum_diff_before + each_block[i]
                else: ## after
                    is_over_after = 1
                    sum_diff_after = sum_diff_after + each_block[i]

            #[-6 6]ないのループで over でも underでもないとき
            else :
                if i<=5: ## before
                    sum_diff_before = sum_diff_before + each_block[i]
                else: ## after
                    sum_diff_after = sum_diff_after + each_block[i]

        ## sum_diff_afterは afterのdiffの合計であり、実際に取引した場合の差分をとっている
        ## 検出時に購入し、5経過したあとにはいくらの差分が開いているのかを確認するため
        close_diff.append( round(sum_diff_after, 3) )        
        open_diff.append( round(sum_diff_before, 3) )  

        ## beforeのクラス分け
        ## 1Sampleあたりにつき複数のdiffブロックがあるので、一つごとに
        if (is_under_before==0) and (is_over_before==0):
            classified_each_blocks_before.append('F')
        elif (is_under_before==0) and (is_over_before == 1):
            classified_each_blocks_before.append('U')
        elif (is_under_before==1) and (is_over_before == 0):
            classified_each_blocks_before.append('D')
        else:
            classified_each_blocks_before.append('V')

        ## afterのクラス分け
        ## 1Sampleあたりにつき複数のdiffブロックがあるので、一つごとに
        if (is_under_after==0) and (is_over_after==0):
            classified_each_blocks_after.append('F')
        elif (is_under_after==0) and (is_over_after==1):
            classified_each_blocks_after.append('U')
        elif (is_under_after==1) and (is_over_after==0):
            classified_each_blocks_after.append('D')
        else:
            classified_each_blocks_after.append("V")

        under_indexs_SAMPLE.append(under_indexs_each)
        over_indexs_SAMPLE.append(over_indexs_each)
        
        # 1つのSAMPLE群の eachごとにリセットする
        under_indexs_each = []
        over_indexs_each = []

        is_under_before = 0
        is_under_after = 0
        is_over_before =0
        is_over_after =0
        sum_diff_before = 0
        sum_diff_after = 0

    ## open_diff と close_diffに対して、カテゴリー分けを行う
    for i in open_diff:
        if i < (diff_std*(-1)):
            classified_diff_before.append("D_diff")
        elif i > diff_std:
            classified_diff_before.append("U_diff")
        elif (diff_std*(-1)) < i < (DIFF_LIMIT*(-1)):
            classified_diff_before.append("D_little_diff")
        elif DIFF_LIMIT < i < diff_std:
            classified_diff_before.append("U_little_diff")
        else:
            classified_diff_before.append("F_diff")

    for i in close_diff:
        if i < (diff_std*(-1)):
            classified_diff_after.append("D_diff")
        elif i > diff_std:
            classified_diff_after.append("U_diff")
        elif (diff_std*(-1)) < i < (DIFF_LIMIT*(-1)):
            classified_diff_after.append("D_little_diff")
        elif DIFF_LIMIT < i < diff_std:
            classified_diff_after.append("U_little_diff")
        else:
            classified_diff_after.append("F_diff")

    print("before" ,classified_each_blocks_before)
    print("after" ,classified_each_blocks_after)
    print("opendiff" ,open_diff)
    print("closediff" ,close_diff)
    print("opendiff_class", classified_diff_before)
    print("closediff_class", classified_diff_after)
    print("std", diff_std)
    print("break")

    return classified_each_blocks_before , classified_each_blocks_after, close_diff

##############################################################
###     カテゴリデータフレームに要素にcolumnsをそのまま追加していく  ###
##############################################################

def get_pandasDF_for_train(candle_mean, candle_std, diff_mean, diff_std, DF_train):

    tmp_se = pd.Series( [candle_mean, candle_std, diff_mean, diff_std], index=DF_train.columns )
    DF_train = DF_train.append( tmp_se, ignore_index=True )
    
    return DF_train


###################################################################################
###################################################################################
###           　　　　　　　　            main文                                   ###
###################################################################################
###################################################################################

##############################################################
###                   Fullのサンプルブロックを作る             ###
##############################################################

#たまには絶対パス指定
df = pd.read_csv('/Users/apple/python/oanda/input_USDJPY_CSV/USDJPY_10m.csv',index_col=0)

df_len = len(df)

### パラメータ設定 ###
SIZE_SIGMA_DIFF = 1
SAMPLE_SIZE = 100

### 出力DFのカテゴリ名を指定 ###
DF_train = pd.DataFrame( columns=['candle_mean','candle_std','diff_mean','diff_std'] ) #このdataFrameに対して、for文の中でデータを追加していく

final_rate = 0

### すべてのsampleからdiffのnumpy arrayを作成する
np_all_diff = get_diff_1_ALLrate(df)

# 100ごとのsampleブロックから、しきい値を検出し、high_Vola のaroud[-6:6]をgetする
# SAMPLESIZEごとにサンプルを取ってくる。
# DF → DF
sample_blocks = get_full_sample_fromALLrate(df,SAMPLE_SIZE)

#for i in tqdm(range(sample_blocks)):
## 1Sampleごとのループ 1SAMPLEの大きさは100
for i in tqdm(sample_blocks):
    
    ## 1 Sample Blockからしきい値をget
    under_Threshold , over_Threshold , candle_mean , candle_std = get_Theshold(i)
    
    """
    for j,k in zip(highVola_index,open_rate_vola):
        all_index.append(j)
        open_rate_list.append(k)
        diff_list_forDF.append( np_all_diff[j] )
    """

    ## around 6 のshort blocksのdiffを撮ってくる。ついでに、連も撮ってきている
    ## DF to difflist
    #diff_blocks_intense, list_cnt_before_p, list_cnt_before_m, list_cnt_after_p, list_cnt_after_m = get_diff1_from_HIGHVOLA_Blocks(highVola_Blocks)
    np_diff , final_rate = get_diff1_from_SAMPLE_Blocks(i, final_rate)

    ## 1 Sample listから、激しいdiffであるという根拠のしきい値をとる
    ## DF to value
    point_2sigma_under, point_2sigma_over, diff_mean, diff_std = get_Limit_Each_Blocks(np_diff, SIZE_SIGMA_DIFF)

    # pandas DFをmain文で定義して、DFを引数としてpandas作成関数に渡す
    # こうすることによって、違うSAMPLE BLOCKSに対しても同じDFにデータを入れることができる    
    DF_train = get_pandasDF_for_train(candle_mean, candle_std, diff_mean, diff_std, DF_train)

## CSVへと出力
DF_train.to_csv("/Users/apple/python/oanda/output_classified_csv/Statistic_VALUE_fromALL.csv")
