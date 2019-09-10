'''using numpy polyfit and also using sikit learn polynomial features'''
#using numpy
import numpy as np

# create arrays of fake points
x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])

# fit up to deg=3
z = np.polyfit(x, y, 3)
print(z)



#####################################################################
#using sikit-learn
# Import
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Create matrix and vectors
X = [[0.44, 0.68], [0.99, 0.23]]
y = [109.85, 155.72]
#X_test = [0.49, 0.18]

# PolynomialFeatures (prepreprocessing)
poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(X)
#X_test_ = poly.fit_transform(X_test)


# Instantiate
lg = LinearRegression()

# Fit
lg.fit(X_, y)

# Obtain coefficients
print(lg.coef_)

# Predict
#lg.predict(X_test_)





