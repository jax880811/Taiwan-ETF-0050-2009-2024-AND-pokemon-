import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.svm import LinearSVC

'''
SVM（支持向量機） 是一種 監督式學習 演算法，主要用於分類和回歸任務，目標是找出分隔兩個類別的 最佳超平面。

核心概念
超平面：分隔數據的邊界線（2D）或平面（3D+）。
最大化間隔：SVM 尋找能最大化 分隔間隔（Margin） 的超平面，確保分類效果最佳。
支持向量：距離超平面最近的資料點，決定超平面的位置。
核函數（Kernel Function）
當數據非線性可分時，SVM 會透過 核函數 將數據映射到更高維度的空間進行分類：

線性核：Linear
多項式核：Polynomial
高斯核（RBF）：rbf
Sigmoid 核：類似神經網路中的激活函數。
'''

df=pd.read_csv('Pokemon.csv', encoding='cp1252',header=0)

df['hasType2'] = df['Type_2'].notnull().astype(int)

X, y = df.loc[:, 'HP':'Speed'], df['hasType2']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scale = StandardScaler().fit(X_train)
X_train_std = scale.transform(X_train)
X_test_std = scale.transform(X_test)
Linear_svm = LinearSVC(max_iter=1500)
'''
LinearSVC：SVM 使用線性核函數（適用於線性可分的資料）。
max_iter=1500：設置最大迭代次數，確保收斂。
'''
Linear_svm.fit(X_train_std, y_train)

y_pred = Linear_svm.predict(X_test_std)
print(classification_report(y_test, y_pred))
svm = SVC(kernel='sigmoid', C=5,gamma=.01,probability= True,class_weight='balanced')
'''
SVC：另一個 SVM 模型，支援非線性核函數。
kernel='sigmoid'：使用 Sigmoid 核函數，適合非線性數據。
C=5：正則化參數，控制模型的罰則強度（越大罰則越嚴格）。
gamma=0.01：影響訓練資料的權重（對於非線性核函數非常重要）。
class_weight='balanced'：自動調整類別權重，適合不平衡資料。
probability=True：允許預測時輸出機率估計。
'''
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
print(classification_report(y_test, y_pred))


