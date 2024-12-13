import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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