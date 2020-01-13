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

def get_Statistics_sample(sample_set):

    candle_std = sample_set['OPEN'].std()
    candle_mean = sample_set['OPEN'].mean()

    return candle_mean , candle_std


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
df = pd.read_csv('/Users/apple/python/oanda/input_USDJPY_CSV/USDJPY_10m.csv',index_col=0)

df_len = len(df)

##############################################################
###                    パラメータ設定 　　　　　　　            ###
##############################################################

SIZE_SIGMA_DIFF = 1     ## each diffのしきい値を　2sigmaにするか、1sigmaにするか
SAMPLE_SIZE = 100       ## 1サンプルの数
Change_Theshold_rate_sw = 100   ## rateの high volaを検出するときのしきい値
                                ## 0 → 標準偏差*1 
                                ## 1 → 標準偏差*1.64 
                                ## 2 → 標準偏差*1.96 
                                ## 3 → 標準偏差*0.75 
                                ## 4 → 標準偏差*0.50 
                                ## 100 → 0 すなわち、すべてのデータを取ってくる 

diff_sw = 0                     ## diffが 大きいのかどうかを判断するためのdiff
                                ## 0 → relative 1sample(100)ごとに、diffの平均をとって、しきい値を決めている
                                ## 1 → absolute 2001〜2019のすべての平均値を採用

Change_Theshold_diff_sw = 0     ## diffの high volaを検出するときのしきい値
                                ## 0 → 平均 + 標準偏差*1
                                ## 1 → 平均 + 標準偏差*1.64 
                                ## 2 → 平均 + 標準偏差*1.96 
                                ## 3 → 平均 + 標準偏差*0.75 
                                ## 4 → 平均 + 標準偏差*0.50 

## しきい値、基準は別途計算した統計値 ##
DIFF_MEAN_ABSOLUTE = 0.051887    ## all diff　すべてのサンプル2001~2019でのdiffの平均



### 出力DFのカテゴリ名を指定 ###
DF_train = pd.DataFrame( columns=['extreme_point','category_Before','category_After','diff_open','diff_close','classdiff_before','classdiff_after','SeqBefore_p','SeqBefore_m', 'SeqAfter_p', 'SeqAfter_m'] ) #このdataFrameに対して、for文の中でデータを追加していく

all_index = []         # for input data to pandas すべてのサンプルのindexリスト 
open_rate_list = []    # for input data to pandas 価格リスト
diff_list_forDF = []   # for input data to pandas 一回差分

### すべてのsampleからdiffのnumpy arrayを作成する
np_all_diff = get_diff_1_ALLrate(df)

# 100ごとのsampleブロックから、しきい値を検出し、high_Vola のaroud[-6:6]をgetする
# SAMPLESIZEごとにサンプルを取ってくる。
# DF → DF
sample_blocks = get_full_sample_fromALLrate(df,SAMPLE_SIZE)

#for i in tqdm(range(sample_blocks)):
## 1Sampleごとのループ 1SAMPLEの大きさは100
for i in tqdm(sample_blocks):
    
    ## 1 Sample Blockから、rateの 平均/分散を取得
    ## configによって、high volaを検出するしきい値を変更する
    candle_mean , candle_std = get_Statistics_sample(i)

    if( Change_Theshold_rate_sw==0 ):
        std_line = candle_std
        Name_rate_sw = 'RATE_THRESHOLD1_0'
    elif( Change_Theshold_rate_sw==1 ):
        std_line = 1.64*candle_std
        Name_rate_sw = 'RATE_THRESHOLD1_64'
    elif( Change_Theshold_rate_sw==2 ):
        std_line = 1.96*candle_std
        Name_rate_sw = 'RATE_THRESHOLD1_96'
    elif( Change_Theshold_rate_sw==3 ):
        std_line = 0.75*candle_std
        Name_rate_sw = 'RATE_THRESHOLD0_75'
    elif( Change_Theshold_rate_sw==4 ):
        std_line = 0.50*candle_std
        Name_rate_sw = 'RATE_THRESHOLD0_50'
    elif( Change_Theshold_rate_sw==100 ):
        std_line = 0*candle_std
        Name_rate_sw = 'RATE_THRESHOLD_NONE'
    else:
        None

    under_Threshold = candle_mean - std_line
    over_Threshold = candle_mean + std_line


    ## 1 Sample Blockから、しきい値超えのINDEXをget 連続値は省き、aroudはまだ
    ## DF to list(index)
    highVola_index , open_rate_vola , extreme_point = get_highVolatility_index(i, under_Threshold , over_Threshold)


    ## Dfに rateと、diffを追加
    for j,k in zip(highVola_index,open_rate_vola):
        all_index.append(j)
        open_rate_list.append(k)
        diff_list_forDF.append( np_all_diff[j] )

    # 分散が大きい大きいindexが存在しなかったらパスする
    if len(highVola_index) == 0:
        None
    else:
        ## 連続indexをdrop済みの、INDEXをもとに、around 6 を拾ってくる[-6:6] Dfのまんま
        ## DF to DF
        highVola_Blocks = get_high_vola_Blocks(df, highVola_index)
        
        ## around 6 のshort blocksのdiffを撮ってくる。ついでに、連も撮ってきている
        ## DF to difflist
        diff_blocks_intense, list_cnt_before_p, list_cnt_before_m, list_cnt_after_p, list_cnt_after_m = get_diff1_from_HIGHVOLA_Blocks(highVola_Blocks)
        
        ## 1 Sample listから、激しいdiffであるという根拠のしきい値をとる
        ## DF to value
        diff_mean_each, diff_relative = get_diff_Limit_Each_Blocks(i)

        ''' diffに関するチューニング '''
        ## Relativeにするか、ABSOLUTEにするか
        ## 出力ファイルの名前も
        if (diff_sw == 0):
            diff_limit = diff_relative
            Name_diff_sw = 'RELATIVE_DIFF_THESHOLD_'
        elif (diff_sw == 1):
            diff_limit = DIFF_MEAN_ABSOLUTE        
            Name_diff_sw = 'ABSOLUTE_DIFF_THESHOLD_'
        else:
            None

        ## diffのFlatと判定するしきい値のチューニング
        ## 出力ファイルの名前も
        if Change_Theshold_diff_sw == 0: 
            Name_std_config = '1_0'
        elif Change_Theshold_diff_sw == 1:
            diff_limit = 1.64*diff_limit
            Name_std_config = '1_64'
        elif Change_Theshold_diff_sw == 2:
            diff_limit = 1.96*diff_limit
            Name_std_config = '1_96'
        elif Change_Theshold_diff_sw == 3:
            diff_limit = 0.75*diff_limit
            Name_std_config = '0_75'
        elif Change_Theshold_diff_sw == 4:
            diff_limit = 0.50*diff_limit
            Name_std_config = '0_50'
        else:
            None

        classified_each_blocks_before, classified_each_blocks_after, open_diff, close_diff,classified_diff_before,classified_diff_after = get_Categoly_diff(diff_limit, diff_blocks_intense)
    
        # pandas DFをmain文で定義して、DFを引数としてpandas作成関数に渡す
        # こうすることによって、違うSAMPLE BLOCKSに対しても同じDFにデータを入れることができる
        
        DF_train = get_pandasDF_for_train(diff_blocks_intense,
                                          extreme_point,
                                          classified_each_blocks_before,
                                          classified_each_blocks_after,
                                          open_diff,
                                          close_diff,
                                          classified_diff_before,
                                          classified_diff_after,
                                          list_cnt_before_p,
                                          list_cnt_before_m,
                                          list_cnt_after_p,
                                          list_cnt_after_m,
                                          DF_train)

# カテゴリー分けを行ったpandasDFに対して、indexとrateの列を追加する
# 必要はないが念の為
tmp_se_index = pd.Series( all_index )
tmp_se_rate = pd.Series( open_rate_list )
tmp_se_diff = pd.Series( diff_list_forDF )
DF_train["INDEX_ALL_SAMPLE"]=tmp_se_index
DF_train["RATE"]=tmp_se_rate
DF_train["DIFF"]=tmp_se_diff

print(len(DF_train))
print(DF_train.head())



################################################################
###                   出力ファイルの名前                        ###
################################################################

csvfile_name_classified = 'classified_sample_' + Name_rate_sw + '_' + Name_diff_sw + Name_std_config

DF_train.to_csv("/Users/apple/python/oanda/output_classified_csv/test_variation/%s.csv" % csvfile_name_classified)

# CSVへと出力
#DF_train.to_csv("/Users/apple/python/oanda/output_classified_csv/%s_after2014.csv" % csvfile_name_classified)
#DF_train.to_csv("/Users/apple/python/oanda/output_classified_csv/%s_ALL_TRUE_THESHOLD.csv" % csvfile_name_classified)
#DF_train.to_csv("/Users/apple/python/oanda/output_classified_csv/%s_ALL_OPEN_CLOSE_THESHOLD.csv" % csvfile_name_classified)
