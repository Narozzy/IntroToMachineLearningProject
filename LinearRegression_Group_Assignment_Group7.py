import numpy as np
import matplotlib as mpl
import sklearn as sk
import sklearn.linear_model as sklm
import sklearn.model_selection as skms

"""
Function combines 2D arrays and adds the 1 at the end
This is so we can use the array in multivar regression.
"""
def combine_2D_arrays(a1, a2):
    if len(a1) != len(a2):
        raise Exception("Parameters a1 and a2 must be same size")
    
    l = list()
    for i in range(len(a1)):
        l.append(np.array([a1[i], a2[i], 1]))
    return np.array(l)

### Data from .csv stored in here
    ### Originally had the absolute address of PubgData.csv in this function
    ### hopefully this will load if not I apologize
data = np.genfromtxt("PubgData.csv", delimiter=',', skip_header=1)

### Data selected and converted into NumPy arrays
train_x = np.copy(data[:10000, 3]).reshape(-1, 1)      ### dmgDealt column
test_x = np.copy(data[10000:20000, 3]).reshape(-1, 1)  ### dmgDealt column

train_y = np.copy(data[:10000, -1]).reshape(-1, 1)     ### winPercent
test_y = np.copy(data[10000:20000, -1]).reshape(-1, 1) ### winPercent

### Create our linear regression model and fit it to our training set
lin_reg = sklm.LinearRegression()
lin_reg.fit(train_x, train_y)

### Plot the testing points and the line formed from lin_reg model
predict_y = lin_reg.predict(test_x)
mpl.pyplot.scatter(test_x, test_y, color='black')
mpl.pyplot.plot(test_x, predict_y, color='red')

### Print the mean squared error of the linear regression model
print("Mean Squared Error of the Linear Regression Model: %.5f"
      % sk.metrics.mean_squared_error(test_y, predict_y))


"""
K-Fold Cross Validation
"""
kf = skms.KFold(n_splits=5)

### Perform k=5 Cross Validation
for train_index, test_index in kf.split(train_x):
    print("TRAIN: ", train_index, "TEST: ", test_index)
    x_training, x_testing = train_x[train_index], test_x[test_index]
    y_training, y_testing = train_y[train_index], test_y[test_index]

### Fit our linear regression model to our cross-validation training set    
CV_lin_reg = sklm.LinearRegression()
CV_lin_reg.fit(x_training, y_training)

### Plot the testing points and the line formed from lin_reg model w/ CV training set
CV_predict_y = CV_lin_reg.predict(x_testing)
mpl.pyplot.scatter(x_testing, y_testing, color='black')
mpl.pyplot.plot(x_testing, CV_predict_y, color='red')

# Print the mean squared error of the linear regression model w/ CV data set
print("Mean Squared Error of the Cross-Validation Linear Regression Model: %.5f"
      % sk.metrics.mean_squared_error(y_testing, CV_predict_y))



"""
Multivariate Training sets
"""
train_x2 = np.copy(data[:10000, 5]).reshape(-1, 1)     ### killPlace column (train)
test_x2 = np.copy(data[10000:20000, 5]).reshape(-1, 1) ### killPlace column (test)

### Using helper function combine_2D_arrays to set up multivar regression numpy arrays
multivar_training_x = combine_2D_arrays(train_x, train_x2)
multivar_testing_x = combine_2D_arrays(test_x, test_x2)

### Fit our model with a tuple containing two qualities of data set and train_y
multivar_reg = sklm.LinearRegression()
multivar_reg.fit(multivar_training_x, train_y)

multivar_predict_y = multivar_reg.predict(multivar_training_x)

p = np.array(multivar_reg.predict(multivar_training_x))

### Calculate the relative error of the regression model
err = abs(p - test_y)

### Calculate the total_error of our model by using the dot product of our relative error
total_error = np.dot(np.squeeze(err), np.squeeze(err))

rmse_train = np.sqrt(total_error/len(p))

### Create our plot window
mpl.pyplot.plot(p, test_y, 'ro')
mpl.pyplot.plot([0, 2500],[0,5.0], 'g-')
mpl.pyplot.xlabel('Property Estimates')
mpl.pyplot.ylabel('winPlacePrec')
### Show our plot window
mpl.pyplot.show()

### Print values associated with the model
print("Regression Coefficients: ",  multivar_reg.coef_)
print("RMSError of the regression model: %.5f" % rmse_train)
print("Mean Squared Error of the Cross Validation Multivariate Linear Regression: %.5f\n"
      % sk.metrics.mean_squared_error(test_y, multivar_predict_y))


"""
KFold of the Multivariate Regression
"""
kf = skms.KFold(n_splits=5)
error_wrt_x = 0

### Perform k=5 Cross Validation
for train, test in kf.split(multivar_training_x):
    multivar_reg.fit(multivar_training_x[train], train_y[train])
    p = multivar_reg.predict(multivar_training_x[test])
    err = p - test_y[test]
    error_wrt_x += np.dot(np.squeeze(err), np.squeeze(err))

### Calculate the RMSE of this k=5 Cross Validation
RMSE_5_CV = np.sqrt(error_wrt_x/len(multivar_training_x))

### Print the regression coefficients for multivar_reg w/ k-Fold value of 5
print('Regression Coefficients: ', multivar_reg.coef_)

### Compare the two RMSE we have obtained
print('RMSE on training: %.5f' % rmse_train)
print('RMSE on 5-split Cross Validation, using values damageDealt and killPlace: %.5f' % RMSE_5_CV)
