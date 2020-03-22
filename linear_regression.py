from sklearn import datasets, linear_model

import numpy as np
import matplotlib.pyplot as plt

#Loading the dataset.
digits = datasets.load_digits()

digits_X = digits.data[:,np.newaxis,2]

#Training data.
digits_X_train = digits_X[:-30]
#Test data.
digits_X_test = digits_X[-30:]
digits_y_train = digits.target[:-30]
digits_y_test = digits.target[-30:]

#Performing linear regression.
regr = linear_model.LinearRegression()
regr.fit(digits_X_train,digits_y_train)

#Defining data to be predicted upon.
digits_y_pred = regr.predict(digits_X_test)
print(regr.coef_)

#Plotting scatter plot between test data.
plt.scatter(digits_X_test,digits_y_test,color = 'blue')

plt.plot(digits_X_test,digits_y_pred,color = 'red',linewidth = 3)
plt.xticks(())
plt.yticks(())
plt.show()