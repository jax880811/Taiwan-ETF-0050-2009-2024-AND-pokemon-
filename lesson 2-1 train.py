import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

df=pd.read_csv('final.csv', encoding='cp1252',header=0)
#df = df.fillna(df.mean()) #填補缺失的值

plt.style.use('fivethirtyeight')
x,y = df.loc[:,['adj close']],df.loc[:,['open']]
lr = LinearRegression()
lr.fit(x,y)
#做一元線性回歸
print('w_1 = ',lr.coef_[0]) 
print('w_0 = ',lr.intercept_)

plt.scatter(x,y,facecolor = 'xkcd:azure',edgecolor = 'black',s = 20)
plt.xlabel('adj close')
plt.ylabel('open')
sns.regplot(x = 'high',y = 'close',data = df ,scatter_kws={'facecolor' : 'xkcd:azure','edgecolor' : 'black','s':20},line_kws={'color':'r','lw':3})

plt.show()

x,y = df.loc[:,['high','low']],df.loc[:,['open']] #依照開盤當日的最高以及最低，預測開盤價位
x = sm.add_constant(x)
model = sm.OLS(y,x)
result = model.fit()
print('迴歸係數:',result.params)
print(result.summary())
