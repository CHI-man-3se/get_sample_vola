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
    rand_index = random.randint(0, df_len-sample_size)

    date = df.at[rand_index, 'DTYYYYMMDD']

    sample_set = df.iloc[rand_index:rand_index+sample_size,:]  
    
    return sample_set ,date , rand_index


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
###           rate ( candle ) の平均、分散を取得する           ###
##############################################################

def get_Statistics_sample_candle(sample_set):

    candle_std = sample_set['OPEN'].std()
    candle_mean = sample_set['OPEN'].mean()

    return candle_mean , candle_std

##############################################################
###                diff の平均、分散を取得する                 ###
##############################################################

def get_Statistics_sample_diff(sample_set):

    diff_std = sample_set['DIFF'].std()
    diff_mean = sample_set['DIFF'].mean()

    return diff_mean , diff_std




##############################################################
###           分散が大きいindex番号だけをとってくる              ###
##############################################################

def get_highVolatility_index(sample_set ,under_Threshold, over_Threshold):

    ## highVolaの検出なしで、すべてのindexを持ってきたいときdropアルゴリズムによって終わる
    ## なので、debug用としてあらたなdropアルゴリズムのパスをつくる
    debug_ALL = 0
    if under_Threshold==over_Threshold:
        debug_ALL = 1
    else:
        debug_ALL = 0

    # high_volaのdataframeをゲット　※ほしいのはindexだけなので、それだけ貰えればいいかも
    high_volatility = sample_set.query('OPEN <= @under_Threshold or @over_Threshold <= OPEN')

    sample_len = len(high_volatility)
    
    index_num = high_volatility.index.values
    #open_rate = high_volatility.loc[:,'OPEN']

    open_rate = []
    drop_sequence_index = []
    extreme_point = []

    # 分散がでかくなった瞬間を持ってきたいので、連続indexが連続になっているのは省く
    # indexが連続しているのをdropさせるloop
    if debug_ALL == 0:    
        if sample_len == 0 :
            None
        else:
            
            if 0<=index_num[0]<=6 : ## high vola検出のindexが6以下のときはeach blocksを形成できないためここもパスする
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
    elif debug_ALL == 1:
        index_debug = np.arange(6, 97, 12)
        for i in index_debug:
            drop_sequence_index.append(index_num[i])
    else:
        None

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
###                [-6:6]のブロックの1階差を取る               ###
###     inputのblocksがpandasを要素とするlistなので            ###
###     　　　　　for 文をつかって要素に分解してあげる必要がある    ###
##############################################################
### 引数 DataFrame [-6 6]の　つまり、indexやrate dateすべて含まれている
### 返り値
def get_diff1_from_HIGHVOLA_Blocks(blocks):

    diff_each_blocks = []
    list_cnt_before_p = []
    list_cnt_before_m = []
    list_cnt_after_p = []
    list_cnt_after_m = []

    if (type(blocks) == list):
        for i in blocks:

            # 1階差のdiffをlistに入れる。　1つのsampleブロックにつき、何回激しい分散が来ても対応できるように
            np_blocks = i.iloc[:,2].values
            #np_blocks = i.loc[:,'OPEN'].values
            diff_each_blocks.append( np.diff(np_blocks) )
            
            cnt_before_p, cnt_before_m, cnt_after_p, cnt_after_m = get_Sequencial( np.diff(np_blocks) )
        
            list_cnt_before_p.append(cnt_before_p)
            list_cnt_before_m.append(cnt_before_m)
            list_cnt_after_p.append(cnt_after_p)
            list_cnt_after_m.append(cnt_after_m)

    else:
        None

    return diff_each_blocks, list_cnt_before_p, list_cnt_before_m, list_cnt_after_p, list_cnt_after_m 



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
    np_blocks = blocks.iloc[:,4].values
    tmp_np_diff = np.diff(np_blocks)
    np_diff = np.round(tmp_np_diff, 3)
    diff_each_blocks = np_diff.tolist()
    diff_each_blocks.insert(0, 0)

    return diff_each_blocks

############################################################
###          diffを見て、それがintense or 緩やかを確認したい   ###
###     sample 100ごとの diff の平均をベースに、しきい値を取る   ###
#############################################################

def get_diff_Limit_Each_Blocks(sample_set):

    np_sample = sample_set.iloc[:,2].values
    np_diff_sample = np.diff(np_sample)
    
    diff_std_each = np_diff_sample.std()
    diff_mean_each = np_diff_sample.mean()

    diff_limit_relative = round(diff_std_each, 4)

    return diff_mean_each, diff_limit_relative 

############################################################
###          numpyを使わず普通のリストで行う                  ###
###          diffを見て、それがintense or 緩やかを確認したい   ###
###     sample 100ごとの diff　listを作ってそれのstdをとるか   ###
#############################################################

def get_Categoly_diff(diff_limit, diff_blocks):

    is_under_before = 0
    is_under_after = 0
    is_over_before = 0
    is_over_after = 0

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

    DIFF_LIMIT_LITTLE = (diff_limit/2)   ## little diff用

    ## [-6 6]のdiff blockが複数個あるのでまずはそれを取り出すfor文
    for each_block in diff_blocks:

        #[-6 6]の中を回すfor文
        for i in range(len(each_block)):

            #[-6 6]の中から、under Thresholdを検出しに行く
            if each_block[i] < diff_limit*(-1):
                under_indexs_each.append(i)
                if i<=5: ## before
                    is_under_before = 1
                    sum_diff_before = sum_diff_before + each_block[i]
                else: ## after
                    is_under_after = 1
                    sum_diff_after = sum_diff_after + each_block[i]

            #[-6 6]の中から over Thresholdを検出しに行く
            elif each_block[i] > diff_limit:
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
        if i < (diff_limit*(-1)):
            classified_diff_before.append("D_diff")
        elif i > diff_limit:
            classified_diff_before.append("U_diff")
        elif (diff_limit*(-1)) < i < (DIFF_LIMIT_LITTLE*(-1)):
            classified_diff_before.append("D_little_diff")
        elif DIFF_LIMIT_LITTLE < i < diff_limit:
            classified_diff_before.append("U_little_diff")
        else:
            classified_diff_before.append("F_diff")

    for i in close_diff:
        if i < (diff_limit*(-1)):
            classified_diff_after.append("D_diff")
        elif i > diff_limit:
            classified_diff_after.append("U_diff")
        elif (diff_limit*(-1)) < i < (DIFF_LIMIT_LITTLE*(-1)):
            classified_diff_after.append("D_little_diff")
        elif DIFF_LIMIT_LITTLE < i < diff_limit:
            classified_diff_after.append("U_little_diff")
        else:
            classified_diff_after.append("F_diff")

    return classified_each_blocks_before , classified_each_blocks_after, open_diff ,close_diff, classified_diff_before, classified_diff_after

##############################################################
###     カテゴリデータフレームに要素にcolumnsをそのまま追加していく  ###
##############################################################

def get_pandasDF_for_train(diff_blocks,
                           extreme_point,
                           classified_each_blocks_before,
                           classified_each_blocks_after,
                           open_diff,
                           close_diff,
                           classified_diff_before,
                           classified_diff_after,
                           cnt_before_p,
                           cnt_before_m,
                           cnt_after_p,
                           cnt_after_m,
                           DF_train):

    for i,j,k,l,m,n,o,p,q,r,s in zip(extreme_point,
                                 classified_each_blocks_before,
                                 classified_each_blocks_after,
                                 open_diff,
                                 close_diff,
                                 classified_diff_before,
                                 classified_diff_after,
                                 cnt_before_p,
                                 cnt_before_m,
                                 cnt_after_p,
                                 cnt_after_m):

        tmp_se = pd.Series( [i,j,k,l,m,n,o,p,q,r,s], index=DF_train.columns )
        DF_train = DF_train.append( tmp_se, ignore_index=True )

    return DF_train




#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

def get_Threshold(mean ,std, size):

    under_Threshold = mean - (std*size)
    over_Threshold = mean + (std*size)

    return under_Threshold , over_Threshold


def judge_LatestRate_UseDiff(df, targetpoint, rand_index, sample_size, under_diff_Threshold, over_diff_Threshold):

    under_diff_Threshold = diff_std*(-1)*SIZE_THRESHOLD
    over_diff_Threshold = diff_std*SIZE_THRESHOLD

    start_index = rand_index + SAMPLE_SIZE + targetpoint
    rate_latest = df.at[start_index, 'OPEN']
    diff_latest = df.at[start_index, 'DIFF']

    if(diff_latest <= under_diff_Threshold):
        ret = 'UNDER'
    elif(diff_latest >= over_diff_Threshold):
        ret = 'OVER'
    else:
        ret = 'NONE'

    return ret, start_index, diff_latest, rate_latest

def judge_LatestRate_UseRate(df, targetpoint, rand_index, sample_size, under_rate_Threshold, over_rate_Threshold):

    
    start_index = rand_index + SAMPLE_SIZE
    now_index = rand_index + SAMPLE_SIZE + targetpoint
    rate_latest = df.at[now_index, 'OPEN']

    if(rate_latest <= under_rate_Threshold):
        ret = 'UNDER'
    elif(rate_latest >= over_rate_Threshold):
        ret = 'OVER'
    else:
        ret = 'NONE'

    return ret, start_index, now_index ,rate_latest



def get_diff_after_detect(df, result_judged_rate, detected_index, detected_rate, diff_std):

    under_diff_Threshold = diff_std*(-1)
    over_diff_Threshold = diff_std

    rate_after_list = []
    diff_after_list = []
    result_list = []
    for i in range(1,7):
        rate_after = df.at[detected_index+i, 'OPEN']
        rate_after_list.append( rate_after )

        diff = rate_after - detected_rate 
        diff_after_list.append( round(diff,3) )

        ## latestを一つづつ進めていき、OVERを検出したとき
        if(result_judged_rate == 'OVER'):
            if( diff < under_diff_Threshold):
                ret = 'WIN'
            elif( diff > over_diff_Threshold):
                ret = 'LOOSE'
            elif( under_diff_Threshold <= diff < 0 ):
                ret = 'WIN_LITTLE'
            elif( 0 < diff <  over_diff_Threshold):
                ret = 'LOOSE_LITTLE'
            elif( diff == 0 ):
                ret = 'NOTHING'
            else:
                ret = 'WTF'

        ## latestを一つづつ進めていき、UNDERを検出したとき
        elif(result_judged_rate == 'UNDER'):
            if( diff < under_diff_Threshold):
                ret = 'LOOSE'
            elif( diff > over_diff_Threshold):
                ret = 'WIN'
            elif( under_diff_Threshold <= diff < 0 ):
                ret = 'LOOSE_LITTLE'
            elif( 0 < diff <  over_diff_Threshold):
                ret = 'WIN_LITTLE'
            elif( diff == 0 ):
                ret = 'NOTHING'
            else:
                ret = 'WTF'

        result_list.append(ret)

    return rate_after_list, diff_after_list, result_list

def get_ResultData(Result_rate_after_list, Result_diff_after_list, Result_result_list):

    print(Result_result_list)



def for_Result_Statistic(result_list,sum_Result,winlose,category):
    
    for i in range(0,6):
        for j in range(0,5):
            if (result_list[i] == category[j]) :
                sum_Result[i][j] = sum_Result[i][j] + 1

                if(i == 5):
                    if(j==0 or j==1):
                        winlose[0] = winlose[0] + 1
                    elif(j==3 or j==4):
                        winlose[1] = winlose[1] + 1
                    else:
                        None
                else:
                    None

            else:
                None


def drow_Graph_detected(df,rand_index,detected_index):
    RATE = df.loc[ rand_index:detected_index+7 , 'OPEN']
    detected_point = df.at[ detected_index+1 , 'OPEN']
    index_num = np.arange(rand_index, detected_index+8)
    plt.title('%d' %detected_index)
    plt.plot(index_num, RATE)
    plt.plot(detected_index, detected_point,marker='o', markersize=10)
    plt.show()


def drow_Graph_toDetermine_RateOrDiff(df,rand_index,start_index,detected_index_byRate, under_rate_Threshold, over_rate_Threshold):
    
    detected_index = detected_index_byRate

    ###### diffで検出したときの　マーカー
    ###### Rateで検出したときの　マーカー
    ###### サンプル100ので検出したときの　マーカー
    after = 30

    RATE = df.loc[ rand_index:detected_index+after-1 , 'OPEN']
    detected_point = df.at[ detected_index , 'OPEN']
    start_point = df.at[ start_index , 'OPEN']
    index_num = np.arange(rand_index, detected_index+after)

    under_line =  np.full(len(index_num), under_rate_Threshold)
    over_line =  np.full(len(index_num), over_rate_Threshold)

    plt.plot(index_num, under_line,linestyle='dashed',color='green')
    plt.plot(index_num, over_line,linestyle='dashed',color='green')
    plt.plot(index_num, RATE)
    plt.plot(start_index, start_point,marker='o', markersize=10,color='red')
    plt.plot(detected_index, detected_point,marker='o', markersize=10)
    plt.show()

    

###################################################################################
###################################################################################
###           　　　　　　　　            main文                                   ###
###################################################################################
###################################################################################


##############################################################
###                   Fullのサンプルブロックを作る             ###
##############################################################

#たまには絶対パス指定
# index_col = 0　としているのはもともとのCSVにすでにINDEXを持っているため
'''after 2014'''
#df = pd.read_csv('/Users/apple/python/oanda/input_USDJPY_CSV/USDJPY_10m_after2014.csv',index_col=0)
'''debug after 2014'''
#df = pd.read_csv('/Users/apple/python/oanda/input_USDJPY_CSV/for_debug__USDJPY_10m_after2014.csv',index_col=0)
'''ALL'''
df = pd.read_csv('/Users/apple/python/oanda/input_USDJPY_CSV/USDJPY_10m_DIFF.csv',index_col=0)



##############################################################
###                    パラメータ設定 　　　　　　　            ###
##############################################################

SAMPLE_SIZE = 100
SIZE_THRESHOLD = 1.5

Result_rate_after_list = []
Result_diff_after_list = []
Result_result_list = []

category = ['WIN', 'WIN_LITTLE','NOTHING' ,'LOOSE_LITTLE', 'LOOSE' ]
sum_Result = [[0]*5 for i in range(6)] # [0,0,0,0,0]にcategoryのsumが入っており、　外側のlistが、diff,1〜7を意味する
winlose = [0,0]

mean_target_point = 0

for i in range(100):

    sample_block , date , rand_index = get_rand_sample_fromALLrate(df,SAMPLE_SIZE)

    candle_mean , candle_std = get_Statistics_sample_candle(sample_block)
    diff_mean , diff_std = get_Statistics_sample_diff(sample_block)

    ## 検出アルゴリズムはdiffで判定すべきなのか、rateで判定すべきなのかわからないので2つようい
    under_rate_Threshold, over_rate_Threshold = get_Threshold(candle_mean , candle_std , SIZE_THRESHOLD)
    under_diff_Threshold, over_diff_Threshold = get_Threshold(diff_mean , diff_std , SIZE_THRESHOLD)

    ## diffのしきい値を検出するまでの LOOP
    detect_F = 0
    targetpoint = 0
    while(detect_F==0):
        targetpoint = targetpoint + 1

        ## rateのjadgeはdiffで検出したパターンとrateで検出したパターンと2つ用意する
        result_judged_rate_byRate, start_index ,detected_index_byRate, detected_rate_byRate = judge_LatestRate_UseRate(df, targetpoint,rand_index, SAMPLE_SIZE, under_rate_Threshold, over_rate_Threshold)


        ## while loop 判定用　しきい値超えを検出するまで、loopを回す
        
        if( result_judged_rate_byRate=='NONE' ):
            detect_F = 0
        else:
            detect_F = 1
        
    print(rand_index, detected_index_byRate)
    drow_Graph_toDetermine_RateOrDiff(df,rand_index,start_index,detected_index_byRate, under_rate_Threshold, over_rate_Threshold)




    ##############
    ##############
    ##############


    #rate_after_list, diff_after_list, result_list = get_diff_after_detect(df, result_judged_rate, detected_index, detected_rate, diff_std)

    #drow_Graph_detected(df,rand_index,detected_index)

"""
    for_Result_Statistic(result_list,sum_Result,winlose,category) # sum_Resultの中に足し合わせ用のデータを入れていく
    print(result_list)
    
    mean_target_point = mean_target_point + targetpoint

    '''
    print(" ******* LOOP %d *******" %i)
    print("-------->", diff_std)
    print("----------MOMENT-----------   ")
    print(detected_index)
    print(deceted_diff)
    print(detected_rate)
    print("----------RESULT----------")
    print(result_judged_rate)
    print(rate_after_list)
    print(diff_after_list)
    print(result_list)
    print('\n')
    '''
"""
mean_target_point = round(mean_target_point/100, 3)

print(mean_target_point)
print(sum_Result)
print(winlose)

#get_ResultData(Result_rate_after_list, Result_diff_after_list, Result_result_list)


"""
print(rand_index)
print(date)
print(candle_mean)
print(candle_std)
print(diff_mean)
print(diff_std)
"""

