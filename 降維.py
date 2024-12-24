from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

df=pd.read_csv('final.csv', encoding='cp1252',header=0)

cols = ['adj close','close', 'high', 'low', 'open', 'vwap']
scaler = StandardScaler().fit(df[cols])
cols_std = scaler.transform(df[cols])
df_std = pd.DataFrame(cols_std, columns=cols)
print(df_std.describe())

'''
PCA（主成分分析）
定義：PCA 是一種線性降維技術，通過數據的協方差矩陣提取主要變異方向。
原理：
找到數據的主軸（主成分），這些軸是數據方差最大的方向。
主成分互相正交。
適用場景：
數據量大，特徵之間高度相關時。
'''
num_pc = 3 #CA：建立主成分分析模型，指定提取的主成分數量為 3。
pca = PCA(n_components=num_pc)
pca.fit(df_std)
loadings = pd.DataFrame(pca.components_, columns=cols)
loadings.index = ['PC'+str(i+1) for i in range(num_pc)]
print(loadings)
print('pca.explained_variance:',pca.explained_variance_)#explained_variance_：每個主成分對應的變異量
print('pca.explained_variance_ratio:',pca.explained_variance_ratio_) # 解釋變異比例
plt.style.use('fivethirtyeight')
pc_scores = pd.DataFrame(pca.transform(df_std))
pc_scores.columns = ['PC'+str(i+1) for i in range(num_pc)]
pc_scores.plot(kind='scatter', x='PC1', y='PC2')
plt.show()

'''
KPCA（核主成分分析）
定義：KPCA 是 PCA 的非線性擴展，通過核函數映射數據到高維空間後進行線性降維。
適用場景：數據關係是非線性的。
'''
plt.style.use('fivethirtyeight')
num_pc = 2

kpca = KernelPCA(n_components=num_pc , kernel='rbf',gamma=5)
X_kpca = kpca.fit_transform(df_std)
pc_scores = pd.DataFrame(X_kpca)
pc_scores.columns = ['PC'+str(i+1) for i in range(num_pc)]
pc_scores.plot(kind='scatter', x='PC1', y='PC2')
plt.show()




