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



