import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from itertools import chain


# たまには絶対パス指定
#df = pd.read_csv('/Users/apple/python/oanda/input_USDJPY_CSV/USDJPY_10m_after2014.csv')
df = pd.read_csv('/Users/apple/python/oanda/input_USDJPY_CSV/for_debug__USDJPY_10m_after2014.csv')
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

    return under_Threshold , over_Threshold


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

    for i in drop_sequence_index:
        open_rate.append( high_volatility.loc[i,'OPEN'])


    return drop_sequence_index , open_rate



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
    
    print(diff_block)
    print("cbp",cnt_before_p)
    print("cbm",cnt_before_m)
    print("cap",cnt_after_p)
    print("cam",cnt_after_m)
    print("break")

    return cnt_before_p, cnt_before_m, cnt_after_p, cnt_after_m
    
##############################################################
###           1階差をとり、nparrayに変換する関数                ###
##############################################################

# 引数　pandas
# 返り値　nparray
def get_diff_1_ALLrate(sample):
    
    np_open_array = sample.iloc[:,4].values
    temp_np_open_diff = np.diff(np_open_array)
    np_open_diff = np.round(temp_np_open_diff, 3)
    np_open_diff = np.insert(np_open_diff, 0, 0)

    print(np_open_diff)
    print("break")

    return np_open_diff


##############################################################
###                [-6:6]のブロックの1階差を取る               ###
###     inputのblocksがpandasを要素とするlistなので            ###
###     　　　　　for 文をつかって要素に分解してあげる必要がある    ###
##############################################################
def get_diff1_from_HIGHVOLA_Blocks(blocks):

    diff_each_blocks = []
    list_cnt_before_p = []
    list_cnt_before_m = []
    list_cnt_after_p = []
    list_cnt_after_m = []

    if (type(blocks) == list):
        for i in blocks:
            # 1階差のdiffをlistに入れる。　1つのsampleブロックにつき、何回激しい分散が来ても対応できるように
            np_blocks = i.iloc[:,4].values
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
###     sample 100ごとの diff　listを作ってそれのstdをとるか   ###
#############################################################

def get_Limit_Each_Blocks(sample_set , size_sigma):

    np_sample = sample_set.iloc[:,4].values
    np_diff_sample = np.diff(np_sample)
    
    diff_std = np_diff_sample.std()
    diff_mean = np_diff_sample.mean()

    if size_sigma == 1: 
        point_sigma_over = diff_mean + diff_std
        point_sigma_under = diff_mean - diff_std
    elif size_sigma == 2:
        point_sigma_over = diff_mean + 2*diff_std
        point_sigma_under = diff_mean - 2*diff_std
    else:
        None

    return point_sigma_under , point_sigma_over

############################################################
###          numpyを使わず普通のリストで行う                  ###
###          diffを見て、それがintense or 緩やかを確認したい   ###
###     sample 100ごとの diff　listを作ってそれのstdをとるか   ###
#############################################################

def get_Is_Intense_diff(point_2sigma_under, point_2sigma_over, diff_blocks):

    is_under_before = 0
    is_under_after = 0
    is_over_before = 0
    is_over_after = 0

    classified_each_blocks = []
    classified_each_blocks_before = []
    classified_each_blocks_after = []
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

    under_indexs_each = []      #diffブロック[-6,6]に対してすべて しきい値超えを判定する
    over_indexs_each = []       #diffブロック[-6,6]に対してすべて しきい値超えを判定する
    under_indexs_SAMPLE = []    #上のdiffブロック[-6,6]を1SAMPLEごとに持つ [[-6,6],[-6,6] , ,,]
    over_indexs_SAMPLE = []     #上のdiffブロック[-6,6]を1SAMPLEごとに持つ [[-6,6],[-6,6] , ,,]

    ## [-6 6]のdiff blockが複数個あるのでまずはそれを取り出すfor文
    for each_block in diff_blocks:

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

        """
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
        
        """
        
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

    return classified_each_blocks_before , classified_each_blocks_after

##############################################################
###           　  　　　データを作るために加工する               ###
##############################################################

def get_pandasDF_for_train(diff_blocks, classified_each_blocks_before,classified_each_blocks_after ,cnt_before_p, cnt_before_m, cnt_after_p, cnt_after_m, DF_train):

    for i,j,k,l,m,n in zip(classified_each_blocks_before,classified_each_blocks_after, cnt_before_p, cnt_before_m, cnt_after_p, cnt_after_m):

        tmp_se = pd.Series( [i,j,k,l,m,n], index=DF_train.columns )
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

##############################################################
###                   Fullのサンプルブロックを作る             ###
##############################################################

SIZE_SIGMA_DIFF = 1
SAMPLE_SIZE = 100

csvfile_name_classified = 'classified_sample_' + str(SAMPLE_SIZE)

DF_train = pd.DataFrame( columns=['category_Before','category_After','SeqBefore_p', 'SeqBefore_m', 'SeqAfter_p', 'SeqAfter_m', ] ) #このdataFrameに対して、for文の中でデータを追加していく

all_index = []         # for input data to pandas すべてのサンプルのindexリスト 
open_rate_list = []    # for input data to pandas 価格リスト
diff_list_forDF = []   # for input data to pandas 一回差分

###すべてのsampleからdiffのnumpy arrayを作成する
print(df_len)
np_all_diff = get_diff_1_ALLrate(df)

# 100ごとのsampleブロックから、しきい値を検出し、high_Vola のaroud[-6:6]をgetする
#SAMPLESIZEごとにサンプルを取ってくる。
# DF → DF
sample_blocks = get_full_sample_fromALLrate(df,SAMPLE_SIZE)


for i in sample_blocks:

    under_Threshold , over_Threshold = get_Theshold(i)
    
    highVola_index , open_rate_vola = get_highVolatility_index(i , under_Threshold , over_Threshold)
   
    for j,k in zip(highVola_index,open_rate_vola):
        all_index.append(j)
        open_rate_list.append(k)
        diff_list_forDF.append( np_all_diff[j] )

    # 分散が大きい大きいindexが存在しなかったらパスする
    if len(highVola_index) == 0:
        None
    else:
    
        highVola_Blocks = get_high_vola_Blocks(df, highVola_index)
            
        diff_blocks_intense, list_cnt_before_p, list_cnt_before_m, list_cnt_after_p, list_cnt_after_m = get_diff1_from_HIGHVOLA_Blocks(highVola_Blocks)
        
        point_2sigma_under, point_2sigma_over = get_Limit_Each_Blocks(i , SIZE_SIGMA_DIFF)

        classified_each_blocks_before, classified_each_blocks_after = get_Is_Intense_diff(point_2sigma_under, point_2sigma_over, diff_blocks_intense)
    
        # pandas DFをmain文で定義して、DFを引数としてpandas作成関数に渡す
        # こうすることによって、違うSAMPLE BLOCKSに対しても同じDFにデータを入れることができる
        
        DF_train = get_pandasDF_for_train(diff_blocks_intense, classified_each_blocks_before, classified_each_blocks_after, list_cnt_before_p, list_cnt_before_m, list_cnt_after_p, list_cnt_after_m, DF_train)

# カテゴリー分けを行ったpandasDFに対して、indexとrateの列を追加する
# 必要はないが念の為
tmp_se_index = pd.Series( all_index )
tmp_se_rate = pd.Series( open_rate_list )
tmp_se_diff = pd.Series( diff_list_forDF )
DF_train["INDEX_ALL_SAMPLE"]=tmp_se_index
DF_train["RATE"]=tmp_se_rate
DF_train["DIFF"]=tmp_se_diff

print(len(DF_train))
print(DF_train)

# CSVへと出力
#DF_train.to_csv("/Users/apple/python/oanda/output_classified_csv/%s.csv" % csvfile_name_classified)
