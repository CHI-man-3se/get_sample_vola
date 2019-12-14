# high volatility 取得のための、サンプルゲットスクリプト
### このスクリプトは、高い分散になった瞬間を取得しそこから分散前後のカテゴリー分けを行う
* 分散前　→　じわじわ or 下げ or 上げ
* 分散後　→　じわじわ or 下げ or 上げ

## 以下の方針で進める
1. すべてのsampleから、SMPLE_SIZE(100)毎とってくる
1. 1つのSAMPLE_BLOCKごとにThesholdを取得
1. Thresholdから、外れているindexを取得する。ただし、連続している場合は先頭以外をdropすること
1. 取得したindex から、前後 6sampleずつ撮ってくる
1. high vola前後をカテゴリー分け  

|intense before | intense after | plt |
|---|---|---|
| じわ | じわ |  ~~ |
| じわ | 下げ | ~\ |
| じわ | 上げ | ~/ |
| 下げ | じわ |
| 下げ | 下げ |
| 下げ | 上げ |
| 上げ | じわ |
| 上げ | 下げ |
| 上げ | 上げ |

## X と yについて
### input を　intense before と、1 sampleset自体が↗なのか、↘なのか。
### output を intense afterとする
高い分散を観測した後に、どう動くのかを予想したい  
なので、観測前をinputととして、outputを予想できないかという発想