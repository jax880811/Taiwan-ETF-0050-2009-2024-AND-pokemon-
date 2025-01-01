from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

'''
K-Means 的定義與應用
定義:
K-Means 是一種基於距離的非監督式學習分群算法。
核心思想是最小化群內樣本與群中心的距離平方和。
應用:
市場客群分析
醫學數據分群
圖像壓縮與分割
財務風險評估
可調參數:
n_clusters: 分群數。
init: 初始化中心點的方法，如 random 或 k-means++。
max_iter: 最大迭代次數。
tol: 收斂容差。
'''


df=pd.read_csv('Pokemon.csv', encoding='cp1252',header=0)
t1,t2 = 'Bug' , 'Psychic'
df_clf = df[(df['Type_1']==t1) | (df['Type_1']==t2) ]
df_clf = df_clf[['Type_1' , 'Sp_Atk' , 'Sp_Def']] #對'Sp_Atk' , 'Sp_Def'進行分群

#過濾出兩個屬性
df_clf.reset_index(inplace=True)
idx_0 = [df_clf['Type_1']==t1]
idx_1 = [df_clf['Type_1']==t2]

X = df_clf[['Sp_Atk' , 'Sp_Def']]
scaler = StandardScaler().fit(X) #標準化的必要性: K-Means 使用歐幾里得距離進行分群，標準化避免特徵因量綱不同對距離計算的影響。
X_std = scaler.transform(X)
print(X_std[:2,:])
 
km = KMeans(n_clusters=2,init = 'random') #使用 K-Means 進行分群，指定分群數為 2。
y_pred = km.fit_predict(X_std) #執行分群並返回每個樣本的群組標籤。

def plt_scatter(X_std,y_pred,km):
    c1 ,c2 ='red','blue'
    plt.scatter(X_std[y_pred==0,0],
                X_std[y_pred==0,1],
                color = c1 ,edgecolors='k',s=60)
    plt.scatter(X_std[y_pred==1,0],
                X_std[y_pred==1,1],
                color = c2 ,edgecolors='k',s=60)
    if len(X_std[y_pred==0]) < len(X_std[y_pred==1]):
        c1 ,c2 ='blue','red'

    plt.scatter(X_std[idx_0[0],0],
                X_std[idx_0[0],1],
                color = c1 ,marker='^',s=20,label = t1,alpha=.5)
    plt.scatter(X_std[idx_1[0],0],
                X_std[idx_1[0],1],
                color = c2 ,marker='X',s=20,label = t2,alpha=.5)
    plt.scatter(km.cluster_centers_[:,0],
                km.cluster_centers_[:,1],
                 marker='*',s=500,label = 'centroids',c='yellow',edgecolors='black')
    plt.show()

plt_scatter(X_std,y_pred,km)
    
lst_type = ['Fairy','Fighting','Steel','Ice']
df_clf = df[df['Type_1']==lst_type[0]]
for i in range(1,len(lst_type)):
    temp_df = df[df['Type_1'] == lst_type[i]]
    df_clf = pd.concat([df_clf, temp_df], ignore_index=True)
X = df_clf[['Sp_Atk' , 'Sp_Def']]
scaler = StandardScaler().fit(X)
X_std = scaler.transform(X)    
print(X.shape)

lst_dst = []
for i in range(1,21):
    km = KMeans(n_clusters=i)
    km.fit(X_std)
    lst_dst.append(km.inertia_) #使用 inertia_ 指標計算每個分群數對應的群內平方誤差 (WSS)。
    #肘部法則: 尋找曲線中誤差下降趨緩的點作為最佳分群數。
plt.plot(range(1,21),lst_dst,marker='o')
plt.show()





