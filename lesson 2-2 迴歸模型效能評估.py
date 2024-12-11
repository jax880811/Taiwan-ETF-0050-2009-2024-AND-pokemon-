import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.stattools import durbin_watson

df=pd.read_csv('final.csv', encoding='cp1252',header=0)

x,y = df.loc[:,['high','low']],df.loc[:,['open']]
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=0) #前80%做訓練，後20%做測試
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
#求出MSE
print('MSE(training): %.5f, MSE(testing): %.5f' %( 
    mean_squared_error(y_train, y_train_pred), 
    mean_squared_error(y_test, y_test_pred)))
#求出RMSE
print('RMSE(training): %.5f, RMSE(testing): %.5f' %( 
    mean_squared_error(y_train, y_train_pred)**0.5, 
    mean_squared_error(y_test, y_test_pred)**0.5))
#求出R平方
print('R^2(training): %.5f, R^2(testing): %.5f' %( 
    r2_score(y_train, y_train_pred), 
    r2_score(y_test, y_test_pred)))


'''
雖說R平方高代表模型的解釋力比較高
但也並非代表絕對是好的模型
還是要根據資料的特性去做正確的判斷
MSE同樣也是如此
'''
X = sm.add_constant(x)
model = sm.OLS(y,X).fit()

#Shapiro-wilk檢定
stat,p = shapiro(model.resid)
print('Statistics: %.3f , p-value : %.3f' % (stat,p))

#獨立性檢定 若檢定值介於2-4之間就代表獨立 0-2之間則是不獨立
dw = durbin_watson(model.resid)
print('dw : %.3f' % dw)


