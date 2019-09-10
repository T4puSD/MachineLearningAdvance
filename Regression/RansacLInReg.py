import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('housing.csv')
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
print(df.head())

X= df[['RM']].values
y = df[['MEDV']].values

from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

ransac = RANSACRegressor(LinearRegression(),max_trials = 100,min_samples = 50,
                         loss = 'absolute_loss',residual_threshold= 5.0,
                         random_state = 0)
ransac.fit(X,y)
inliner_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inliner_mask)
line_X = np.arange(3,10,1)
line_y_ransac = ransac.predict(line_X[:,np.newaxis])
plt.scatter(X[inliner_mask],y[inliner_mask],c = 'blue',
            edgecolor = 'white',marker= 'o',label = 'Inliers')
plt.scatter(X[outlier_mask],y[outlier_mask],c = 'limegreen',
            edgecolor = 'white',marker = 's',label ='Outliers')
plt.plot(line_X,line_y_ransac,color = 'black',lw = 2)
plt.xlabel('Average number of rooms')
plt.ylabel('price in $1000s')
plt.legend(loc = 'upper left')
plt.show()
print('SLope: %.3f'%ransac.estimator_.coef_[0])
print('Intercept: %.3f'%ransac.estimator_.intercept_)
