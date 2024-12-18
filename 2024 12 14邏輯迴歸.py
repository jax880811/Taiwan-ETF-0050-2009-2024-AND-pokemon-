import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import pair_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

'''
2024 12 14
其實單看這個數據集，使用分類分析並不算特別好
除非這數據集還含有更多的情報，例如:這前50大公司的元素集合(哪幾家公司)，以及產業組成，類型，市值
該數據集要看整個市場走勢的話，最佳情況還是用迴歸分析會更妥當
能夠大概知道長期的0050走勢
雖然台股短期都是會有波動，但是長期放五年到十年基本上都是賺

不過這次是複習，為了方便，就把以前碩一期末作業的Pokemon資料集拿來使用
'''


df=pd.read_csv('Pokemon_new.csv', encoding='cp1252',header=0)

regression = []

print('邏輯迴歸模型')
X, y = df.loc[:, 'Total':'Speed'], df['isLegendary']
X=np.array(X)
y=np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
#d_class_weights = dict(enumerate(class_weights))
logit = LogisticRegression()
logit.fit(X_train, y_train)
logit.score(X_test,y_test)
class_names = ['YES ', 'NO']
disp = ConfusionMatrixDisplay.from_estimator(logit, X_test, y_test, 
                                             display_labels=class_names, 
                                             cmap=plt.cm.Blues)
plt.grid()
plt.show()
print(logit.score(X_test,y_test))
y_pred = logit.predict(X_test)
print(classification_report(y_test, y_pred)) #classification_report：輸出 精確率（Precision）、召回率（Recall）、F1-Score 等指標來評估模型性能
test_score = logit.score(X_test, y_test) * 100
print('邏輯迴歸 ACCURACY = ',test_score,'%')
regression.append(test_score)


