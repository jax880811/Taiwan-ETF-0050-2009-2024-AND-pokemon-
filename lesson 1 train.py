import pandas as pd
from scipy import stats


df=pd.read_csv('final.csv', encoding='cp1252',header=0)
X = df.loc[:,'adj close':'dollar_volume']

corr_matrix = X.corr(method='spearman')

print(corr_matrix)
print(stats.spearmanr(df['close'],df['volume']))