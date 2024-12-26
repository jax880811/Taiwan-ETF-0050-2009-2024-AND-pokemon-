from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

df=pd.read_csv('Pokemon.csv', encoding='cp1252',header=0)
t1,t2 = 'Bug' , 'Psychic'
df_clf = df[(df['Type_1']==t1) | (df['Type_1']==t2) ]
df_clf = df_clf[['Type_1' , 'Sp_Atk' , 'Sp_Def']] #對'Sp_Atk' , 'Sp_Def'進行分群

#過濾出兩個屬性
df_clf.reset_index(inplace=True)
idx_0 = [df_clf['Type_1']==t1]
idx_1 = [df_clf['Type_1']==t2]

X = df_clf[['Sp_Atk' , 'Sp_Def']]
scaler = StandardScaler().fit(X)
X_std = scaler.transform(X)
print(X_std[:2,:])
 
km = KMeans(n_clusters=2,init = 'random')
y_pred = km.fit_predict(X_std)

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
    