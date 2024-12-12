import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso , LassoCV

df=pd.read_csv('final.csv', encoding='cp1252',header=0)
scalar = StandardScaler()

#X = scalar.fit_transform(df.drop(columns=['date']).iloc[:,['adj close','close','high','low','open']])
X = scalar.fit_transform(df.iloc[:,1:6])#取['adj close','close','high','low','open'] 這些欄位
y = scalar.fit_transform(df.loc[:,['volume']])
print(X)
print(y)

'''
脊迴歸（Ridge Regression）： 這是一種常用的線性迴歸正則化方法，通過在成本函數中加入L2正則項，來防止過擬合，提高模型的泛化能力。
L2正規化是把模型裏頭所有的參數都取平方求和
特徵選擇 : 保留所有特徵，但會減小權重
適用場景 : 多重共線性嚴重，但希望保留所有特徵
模型解釋性 : 所有特徵均保留，模型更穩定

'''
'''
小的 α：模型較為靈活，容易過擬合。
大的 α：模型過於簡化，可能欠擬合。
'''
plt.style.use('fivethirtyeight')
alphas = np.logspace(-3,3,50) #使用 NumPy 的 logspace 函數生成 50 個數值，範圍從10的負三次方到10的三次方，產生五十個等間距的對數值

reg_cv = RidgeCV(alphas,store_cv_results=True)
reg_cv.fit(X,y)
print('最佳的alpha值為 : %.3f' % reg_cv.alpha_)

reg = Ridge(alpha=reg_cv.alpha_)
y_pred = reg.fit(X,y).predict(X)

print('MSE = ',mean_squared_error(y,y_pred))

alphas = np.logspace(-5,3,50)
reg_cv2 = LassoCV(alphas=alphas , cv = 10,n_jobs=-1)
reg_cv2.fit(X,y.reshape(-1))
print('Best alpha : %.3f' % reg_cv2.alpha_)
model = Lasso(alpha=reg_cv2.alpha_ , fit_intercept=False) 
model.fit(X,y)
df_coef = pd.DataFrame(data = model.coef_.reshape(1,-1),columns = df.columns[1:6])
print(df_coef)


'''
Lasso L1正規化：把模型裏頭所有的參數都取絕對值。
特徵選擇 : 可以將部分特徵權重縮為零，實現特徵選擇
適用場景 : 特徵數遠大於樣本數，或希望進行特徵選擇
模型解釋性 : 模型更稀疏，容易解釋
'''