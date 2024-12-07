from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

df=pd.read_csv('final.csv', encoding='cp1252',header=0)


def Linear_Regression():#線性迴歸模型: 目標項為Catch_Rate
    plt.style.use('fivethirtyeight')
    x, y = df.loc[:, ['adj close']], df.loc[:, ['close']]
    lr = LinearRegression()
    lr.fit(x, y)
    print('w_1 =', lr.coef_[0])
    print('w_0 =', lr.intercept_)
    X = sm.add_constant(x.to_numpy())
    model = sm.OLS(y, X)
    result = model.fit()
    print('迴歸係數：', result.params)
    print(result.summary())
    Standard = StandardScaler()
    x = Standard.fit_transform(x)
    y = Standard.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=0)
    lr.fit(X_train,y_train)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    lr.fit(X_train, y_train)
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
    X = df.loc[:, 'adj close'].values
    y = df['close'].values

    pol_d = PolynomialFeatures(degree=2)
    X_poly = pol_d.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=0)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)
    

    y_train_resid = y_train_pred - y_train
    y_test_resid = y_test_pred - y_test 

    sns.residplot(x=y_train_pred, y=y_train_resid, lowess=True, 
              color="skyblue", label='Training data', 
              scatter_kws={'s': 25, 'alpha':0.7}, 
              line_kws={'color': 'b', 'lw':2})

    sns.residplot(x=y_test_pred, y=y_test_resid, lowess=True, 
              color="yellowgreen", label='Testing data', 
              scatter_kws={'s': 25, 'marker':'x'}, 
              line_kws={'color': 'g', 'lw':2})
    plt.xlabel('Predicted') #繪製殘差方圖
    plt.legend()
    plt.show()
Linear_Regression()
#print(df.isna())