import pandas as pd
df = pd.read_csv('housing.csv')
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
print(df.head())
##
X = df[['LSTAT']].values
y = df[['MEDV']].values
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
def lin_regplot(X,y,model):
    plt.scatter(X,y,c = 'steelblue',edgecolor = 'white',s = 70)
    y_pred = model.predict(X)
    plt.plot(X,y_pred,color ='black',lw =2)
    return y_pred
tree = DecisionTreeRegressor(max_depth = 3)
tree.fit(X,y)
sort_idx = X.flatten().argsort()
y_pred = lin_regplot(X[sort_idx],y[sort_idx],tree)
plt.xlabel('%lower status of population')
plt.ylabel('price in $100s')
plt.show()

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print('MSE testing: %.3f'%
      (mean_squared_error(y[sort_idx],y_pred))) #mse less is good
print('R^2 testing: %.3f'%(r2_score(y[sort_idx],y_pred)))#r^2 -->1 is good
