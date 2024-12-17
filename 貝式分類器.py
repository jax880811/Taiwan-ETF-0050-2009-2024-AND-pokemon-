import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB

'''
題外話，以前寫論文做差分隱私相關的文獻研究
有曾考慮過使用貝氏定理，判讀使用者過去走過哪些路段，接著會走甚麼路段
來去判斷需要給予合適的privacy budget，不過因為沒有raw data可以使用
最後則是使用了markov decision process
'''

df=pd.read_csv('Pokemon_new.csv', encoding='cp1252',header=0)

X, y = df.loc[:, 'Total':'Speed'], df['isLegendary']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0) #前80%做訓練，後20%做測試

clf = GaussianNB()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(f1_score(y_test,y_pred))


'''
貝氏分類器 是一種基於貝葉斯定理的概率分類方法。它通過計算一個樣本屬於某個類別的概率，將其分到概率最大的那個類別。

樸素貝葉斯 是貝氏分類器家族中的一種，它假設每個特徵之間是相互獨立的。雖然這個假設在現實中可能並不完全成立，但樸素貝葉斯仍然在很多應用中表現良好。

高斯樸素貝葉斯 是一種樸素貝葉斯分類器，它假設每個特徵都服從高斯分布。這使得它特別適用於處理連續型數值特徵的資料。

貝氏分類器的優點：

簡單高效： 模型簡單，訓練速度快。
適用於多分類問題： 可以用於多個類別的分類。
對小樣本數據表現良好： 不需要大量的訓練數據。
貝氏分類器的缺點：

特徵獨立性假設： 在現實世界中，特徵之間往往存在相關性，這會影響模型的準確性。
對先驗概率的敏感性： 模型的性能受到先驗概率的影響。
'''
