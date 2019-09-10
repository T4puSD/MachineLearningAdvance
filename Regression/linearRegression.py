import pandas as pd
import time
#t1 = time.clock()
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data')
#print(df.shape)
#print(df.head())
#df.to_csv('housing.csv',index = False)
df = pd.read_csv('housing.csv')
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
print(df.head())

#pairplot dataexploration
import seaborn as sns
import matplotlib.pyplot as plt
cols = ['LSTAT','INDUS','NOX','RM','MEDV']
#cols = df.columns

sns.pairplot(df[cols],height = 1.5)
plt.tight_layout()
#plt.show()

#heatmap
plt.figure(2)
import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm,cbar = True,annot = True,square = True,fmt = '.2f',
                 yticklabels = cols,
                 xticklabels = cols)
plt.show()

def lin_regplot(X,y,model):
    plt.scatter(X,y,c = 'steelblue',edgecolor = 'white',s = 70)
    plt.plot(X,model.predict(X),color = 'black',lw = 2)
    return None
X = df[['RM']].values
y = df[['MEDV']].values


from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X,y)
print('Slope: %.3f'%slr.coef_[0])
print('Intercept: %.3f'%slr.intercept_)
lin_regplot(X,y,slr)
plt.xlabel('Average number of roooms[RM]')
plt.ylabel('Price in 1000s [MEDV]')
plt.show()

