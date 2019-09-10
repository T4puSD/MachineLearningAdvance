import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#picking up random points
X = np.array([258,270,294,320,342,368,396,446,480,586])[:,np.newaxis]
y = np.array([236.4,234.4,252.8,314.2,298.6,314.2,342.2,360.8,391.2,390.8])

#lr  = LinearRegression()
#pr = LinearRegression()

# PolynomialFeatures (prepreprocessing)
quadratic = PolynomialFeatures(degree  =2)
X_quad = quadratic.fit_transform(X)

##for linear line(same range as random points
lr  = LinearRegression()
lr.fit(X,y)
X_fit = np.arange(250,600,10)[:,np.newaxis]
y_lin_fit = lr.predict(X_fit)

##for polynomial line
pr = LinearRegression()
pr.fit(X_quad,y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

##ploting everything
from matplotlib import pyplot as plt

#ploting random points
plt.scatter(X,y,label = 'training points')
#ploting linear line
plt.plot(X_fit,y_lin_fit,label = 'linear fit',linestyle = '--')
#ploting polynomial line
plt.plot(X_fit,y_quad_fit,label = 'quadratic fit')
plt.legend(loc = 'upper left')
plt.show()

##error calculation
y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print('Training MSE linear: %.3f,quadratic:%.3f'%(mean_squared_error(y,y_lin_pred),
                                                  mean_squared_error(y,y_quad_pred)))
print('Training R^2 linear: %.3f,quadratic:%.3f'%(r2_score(y,y_lin_pred),
                                                  r2_score(y,y_quad_pred)))
