#to implement regression analysis on house dataset including all 13 features as a input and medv as a target
#also check mean**2 error and r**2 error on both training and testing data
import pandas as pd
df = pd.read_csv('housing.csv')
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
print(df.head())

X = df.iloc[:,:-1].values
y = df['MEDV'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)

from sklearn.linear_model import LinearRegression
slr1 = LinearRegression()
slr1.fit(X_train,y_train)
y_train_pred = slr1.predict(X_train)
y_test_pred = slr1.predict(X_test)

from sklearn.metrics import mean_squared_error
t_train = mean_squared_error(y_train,y_train_pred)
t_test = mean_squared_error(y_test,y_test_pred)
print('MSE Train: %.3f, MSE test: %.3f'%(t_train,t_test))

from sklearn.metrics import r2_score
t_train1 = r2_score(y_train,y_train_pred)
t_test1 = r2_score(y_test,y_test_pred)
print('R2 train: %.3f, R2 test:%.3f'%(t_train1,t_test1))
