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

df=pd.read_csv('Pokemon.csv', encoding='cp1252',header=0)

df['hasType2'] = df['Type_2'].notnull().astype(int)

X, y = df.loc[:, 'HP':'Speed'], df['hasType2']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scale = StandardScaler().fit(X_train)
X_train_std = scale.transform(X_train)
X_test_std = scale.transform(X_test)
Linear_svm = LinearSVC(max_iter=1500)
Linear_svm.fit(X_train_std, y_train)

y_pred = Linear_svm.predict(X_test_std)
print(classification_report(y_test, y_pred))
svm = SVC(kernel='sigmoid', C=5,gamma=.01,probability= True,class_weight='balanced')
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
print(classification_report(y_test, y_pred))


