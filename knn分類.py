from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.neighbors import RadiusNeighborsClassifier

df=pd.read_csv('Pokemon.csv', encoding='cp1252',header=0)


X, y = df.loc[:, 'Total':'Speed'], df['isLegendary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
# 特徵標準化
scale = StandardScaler().fit(X_train)
X_train_std = scale.transform(X_train)
X_test_std = scale.transform(X_test)
# 建立最近鄰模型
neighbors = NearestNeighbors(n_neighbors=3).fit(X_train_std)
new_poke = [[600,150, 50, 120, 80, 140, 60]]
new_poke_std = scale.transform(new_poke)
# 取出最近鄰的距離與索引值
dist, idx = neighbors.kneighbors(new_poke_std)
for d, i in enumerate(idx.ravel()):
    print(df.iloc[i, 1], np.array(X_train.iloc[i, :]), 
        '，標準化後的距離 = %.5f'% dist[0][d])
#建立knn分類器，預設k=5
knn = KNeighborsClassifier(n_jobs=-1)
knn.fit(X_train_std, y_train)
# 輸出預測結果
print('預測結果:',knn.predict(new_poke_std))
# 輸出預測結果的機率
print('預測結果的機率:',knn.predict_proba(new_poke_std))
y_pred = knn.predict(X_test_std)
print(classification_report(y_test, y_pred))
test_score = knn.score(X_test_std, y_test) * 100
print('knn ACCURACY = ',test_score,'%')




