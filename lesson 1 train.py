import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

'''
基本欄位
date:
表示交易日期，格式為「YYYY/MM/DD」。此欄位用於描述每一筆交易紀錄的日期。

adj close:
調整後收盤價，考慮了股息與拆股等調整因素，反映投資者實際獲得的價值。

close:
每天的收盤價，即市場在該交易日結束時的價格。

high:
當日最高價，表示股票當天達到的最高交易價格。

low:
當日最低價，表示股票當天達到的最低交易價格。

open:
當日開盤價，表示當天市場開始交易時的價格。

volume:
當天交易的股票總量，表示市場上該股票的活躍度。

技術指標欄位
rsi (Relative Strength Index):
相對強弱指數，用於衡量價格的相對變化，通常用來判斷超買或超賣情況。

atr (Average True Range):
平均真實波幅，衡量市場波動性的指標，計算一定期間內的價格範圍。

adx (Average Directional Index):
平均方向指數，衡量市場趨勢強度的技術指標。

macd (Moving Average Convergence Divergence):
指數移動平均線差離，用來判斷價格走勢的變化趨勢。

均線與成交量
ema_short:
短期指數移動平均線，通常用來追蹤短期價格趨勢。

ema_long:
長期指數移動平均線，用來判斷長期價格走勢。

vwap (Volume Weighted Average Price):
成交量加權平均價，表示一段時間內按成交量加權的平均價格。

隨機震盪指標
fast_k:
隨機震盪指標 %K 的快速版本，反映價格的即時變動情況。

fast_d:
隨機震盪指標 %D 的快速版本，為 %K 的移動平均線。

slow_k:
隨機震盪指標 %K 的慢速版本，用於平滑即時變動的信號。

slow_d:
隨機震盪指標 %D 的慢速版本，為 slow_k 的移動平均線。

波動與布林通道
garman_kl:
Garman-Klass波動率指標，衡量資產價格波動性的特定技術指標。

bb_low (Bollinger Bands Low):
布林通道下軌，用於標示價格的低位範圍。

bb_mid (Bollinger Bands Mid):
布林通道中軌，為價格的簡單移動平均線。

bb_high (Bollinger Bands High):
布林通道上軌，用於標示價格的高位範圍。

財務與流動性指標
dollar_volume:
每日交易的金額總量，計算方式為價格乘以成交量，反映市場流動性。
'''

df=pd.read_csv('final.csv', encoding='cp1252',header=0)
X = df.loc[:,'adj close':'dollar_volume']

corr_matrix = X.corr(method='spearman')

print(corr_matrix)
print(stats.spearmanr(df['close'],df['volume']))

plt.style.use('fivethirtyeight')
plt.figure(figsize=(24,24))
sns.heatmap(df.drop(columns='date').corr(),annot=True)
plt.show()
corr = df.drop(columns='date').loc[:, 'adj close':'dollar_volume'].corr()
print(corr)