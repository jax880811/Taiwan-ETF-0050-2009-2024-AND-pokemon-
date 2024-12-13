import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor #決策樹迴歸
from sklearn.ensemble import RandomForestRegressor

df=pd.read_csv('final.csv', encoding='cp1252',header=0)

def adj_r2(r2,n,k):
    return 1-(n-1)*(1-r2)/(n-k-1)

plt.style.use('fivethirtyeight')
plt.figure(figsize=(24,24))
sns.heatmap(df.drop(columns='date').corr(),annot=True)
plt.show()
corr = df.drop(columns='date').loc[:, 'adj close':'dollar_volume'].corr()
print(corr)



x , y = df[['open']].values , df['vwap'].values
reg = DecisionTreeRegressor(max_depth = 3) #最大深度為3
reg.fit(x,y)
y_pred = reg.predict(x)
r2 = adj_r2(r2_score(y,y_pred),x.shape[0],1)

x_plot = np.linspace(x.min(),x.max(),10).reshape(-1,1)
y_plot = reg.predict(x_plot)
plt.scatter(x,y,label = 'Training points', alpha=.4)
plt.plot(x_plot,y_plot,color = 'black' , lw = 3 , linestyle = '-',label = 'Decision tree regression $R^2 =%.2F$' % r2)

plt.xlabel('開盤')
plt.ylabel('收盤')
plt.legend()
plt.show()

X = df.loc[:, 'adj close':'open']
y = df['close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

reg_random = RandomForestRegressor(n_estimators=100,criterion='friedman_mse',random_state=0,n_jobs=-1) #預設使用100棵樹進行回歸結果的平均
reg_random.fit(X,y)

y_train_pred = reg_random.predict(X_train)
y_test_pred = reg_random.predict(X_test)
r2 = adj_r2(r2_score(y_train,y_train_pred),X.shape[0],X.shape[1])
print('R^2 train : ', r2)
r2 = adj_r2(r2_score(y_test,y_test_pred),X.shape[0],X.shape[1])
print('R^2 test : ', r2)

y_train_resid = y_train_pred - y_train
y_test_resid = y_test_pred - y_test

plt.figure()

sns.residplot(x=X_train['open'],  # 使用特徵值'adj close'
            y=y_train_resid,lowess=True,color="skyblue",label = 'Training data' , 
              scatter_kws = {'s' : 25 , 'alpha' : 0.7},line_kws={'color' : 'b' , 'lw' : 2})
sns.residplot(x=X_train['open'],  # 使用特徵值'adj close'
            y=y_train_resid, lowess=True, 
              color="yellowgreen", label='Testing data', 
              scatter_kws={'s': 25, 'marker':'x'}, 
              line_kws={'color': 'g', 'lw':2})
plt.legend()
plt.xlabel('預測值')
plt.show()