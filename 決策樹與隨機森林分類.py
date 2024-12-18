from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.feature_selection import SelectFromModel

'''
決策樹（Decision Tree）
定義： 決策樹是一種樹狀結構的分類或回歸模型，通過遞迴地將數據集劃分成更小的子集來進行預測。
分類樹：預測離散類別（如 0 和 1）。
回歸樹：預測連續數值（如房價）。
適用範圍：
資料易於解釋、透明，適合需要「邏輯解釋」的場景。
適用於數值型和類別型特徵。
隨機森林（Random Forest）
定義： 隨機森林是多棵決策樹的集合（集成學習方法），通過對多棵樹的結果進行投票或平均來做最終預測。
優勢：
減少決策樹的過擬合問題。
對異常值和缺失值具有較強的容錯能力。
適用範圍：
大型數據集，適合高維度特徵與非線性關係。
分類和回歸任務（如客戶流失預測、銷售預測）。
總結
決策樹 適合簡單邏輯與快速建模，但容易過擬合。
隨機森林 通過集成多棵樹克服過擬合問題，提升模型穩定性和準確率，適合更複雜的場景。
'''

df=pd.read_csv('Pokemon.csv', encoding='cp1252',header=0)
df['hasType2'] = df['Type_2'].notnull().astype(int)

X, y = df.loc[:, 'HP':'Speed'], df['hasType2']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) #預設75%訓練
clf = DecisionTreeClassifier(max_depth=3)#決策樹分類器
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test,y_pred))

plt.figure(dpi=200)
plot_tree(clf,filled=True) #產生一棵樹
plt.show()

random_clf = RandomForestClassifier(max_depth=3, n_jobs=-1) #隨機森林分類器
#n_jobs=-1：利用所有 CPU 核心進行並行運算，加速訓練

random_clf.fit(X_train, y_train)
y_pred = random_clf.predict(X_test)
print(classification_report(y_test, y_pred))
importances = random_clf.feature_importances_
std = np.std([t.feature_importances_ for t in random_clf.estimators_], axis=0)
idx = np.argsort(importances)[::-1]
plt.figure(dpi=200)
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[idx], 
    yerr=std[idx], align="center")
plt.xticks(range(X.shape[1]), labels=X.columns[idx])
plt.xlim([-1, X.shape[1]])
plt.ylim([0, 1])
plt.show()
'''
feature_importances_：
計算每個特徵對模型的重要性。
根據多棵決策樹中，每次分割的貢獻程度計算出平均重要性。
繪圖：
使用柱狀圖顯示特徵重要性，並排序從高到低。
yerr 顯示標準差，反映特徵重要性穩定性。
'''

selector = SelectFromModel(random_clf) #建立特徵選取器，門檻值預設為重要性的平均值
selector.fit(X_train, y_train)
print('門檻值 =', selector.threshold_)
print('特徵遮罩：', selector.get_support())
# 選出新特徵，重新訓練隨機森林
X_train_new = selector.transform(X_train)
random_clf.fit(X_train_new, y_train)
X_test_new = selector.transform(X_test)
y_pred = random_clf.predict(X_test_new)
print(classification_report(y_test, y_pred))


