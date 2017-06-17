import pandas as pd
import numpy as np
def readGermanCreditDataset(fname):
    TrainData = open(fname, 'r')
    featureMap = {'A11':1,'A12':2,'A13':3,'A14':4,
                      'A30':1,'A31':2,'A32':3,'A33':4,'A34':5,
                      'A40':1,'A41':2,'A42':3,'A43':4,'A44':5, 'A45':6,'A46':7,'A47':8,'A48':9,'A49':10, 'A410':0,
                      'A61':2,'A62':3,'A63':4,'A64':5,'A65':1,
                      'A71':1,'A72':2,'A73':3,'A74':4,'A75':5,
                      'A91':1,'A92':4,'A93':2,'A94':3,'A95':4,
                      'A101':1,'A102':2,'A103':3,
                      'A121':4,'A122':3,'A123':2,'A124':1,
                      'A141':1,'A142':2,'A143':3,
                      'A151':2,'A152':3,'A153':1,
                      'A171':1,'A172':2,'A173':3, 'A174':4,
                      'A191':1,'A192':2,
                      'A201':1,'A202':2}
    inp = []
    for line in TrainData:
        inputdata = []
        numdata = []
        for word in line.split():
            if word in featureMap:
                inputdata.append(featureMap[word])
            else:
                 numdata.append(int(word))

        inp.append(inputdata+numdata)
    TrainData.close()
    inp = np.array(inp)
    return inp
XTrain = readGermanCreditDataset("C:\\Users\\priyagoe\\Downloads\\Anomaly-Detection-practical-master\\Anomaly-Detection-practical-master\\Challenges\\German Credit Dataset Classification - Challenge 1\\german.adcg.trainingd")
print('XTrain')
print(XTrain)
XTest = readGermanCreditDataset("C:\\Users\\priyagoe\\Downloads\\Anomaly-Detection-practical-master\\Anomaly-Detection-practical-master\\Challenges\\German Credit Dataset Classification - Challenge 1\\german.adcg.testingd")
print('XTest')
print(XTest)
YTrain = np.loadtxt("C:\\Users\\priyagoe\\Downloads\\Anomaly-Detection-practical-master\\Anomaly-Detection-practical-master\\Challenges\\German Credit Dataset Classification - Challenge 1\\german.adcg.training.label", delimiter=",", skiprows=1, usecols=(1,))
print('YTrain')
print(YTrain)
YTest =  np.loadtxt("C:\\Users\\priyagoe\\Downloads\\Anomaly-Detection-practical-master\\Anomaly-Detection-practical-master\\Challenges\\German Credit Dataset Classification - Challenge 1\\german.adcg.testing.label", delimiter=",", skiprows=1, usecols=(1,))
print('YTest')
print(YTest)
from sklearn.preprocessing import OneHotEncoder
encTrain1 = OneHotEncoder()
encTrain = encTrain1.fit_transform(XTrain[:,:12]).toarray()
print('encTrain')
print(encTrain)

encTest1 = OneHotEncoder()
encTest = encTest1.fit_transform(XTest[:,:12]).toarray()
print('encTest')
print(encTest)

encNTrain = XTrain[:,13:]
encNTest = XTest[:,13:]
#Normalizing & scaling of data
from sklearn import preprocessing
# encNTrain = preprocessing.scale(encNTrain)
encNTrain = (encNTrain - encNTrain.mean())/encNTrain.std()
# encNTest = preprocessing.scale(encNTest)
print('encNTrain')
print(encNTrain)
encNTest = (encNTest - encNTest.mean())/encNTest.std()
print('encNTest')
print(encNTest)
XTrain = np.append(encTrain,encNTrain,axis=1)
XTest = np.append(encTest,encNTest,axis=1)
print('XTrain')
print(XTrain)
print('XTest')
print(XTest)
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

train_error = np.empty(5)
test_error = np.empty(5)

est = make_pipeline(PolynomialFeatures(2), LinearRegression())
est.fit(XTrain, YTrain)
print('expected: train y')
print(YTrain)
print('outcome:')
print(est.predict(XTrain))
train_error = mean_squared_error(YTrain, est.predict(XTrain))
print('expected: test y')
print(YTest)
print('outcome:')
print(est.predict(XTest))
test_error = mean_squared_error(YTest, est.predict(XTest))
print('train error')
print (train_error)
print('test error')
print (test_error)

##LinearRegression
from sklearn import linear_model
from sklearn.decomposition import PCA
pca = PCA(n_components=48)
print (XTrain.shape)
pca.fit(XTrain)
XTrain = pca.transform(XTrain)
print (XTrain.shape)
print (XTest.shape)
pca.fit(XTest)
XTest = pca.transform(XTest)
print (XTest.shape)
LR = linear_model.LinearRegression()
LR.fit(XTrain, YTrain)
LR.predict(XTest)
test_error = mean_squared_error(YTest, LR.predict(XTest))

#Ridge Regression Cross Validation to select optimal value of penalty alpha to reduce overfitting
clfR = linear_model.RidgeCV(alphas=[0.1, 0.3, 0.7, 1.0, 1.3, 1.7, 2, 2.3, 5, 7, 15, 50, 100])
clfR.fit(XTrain, YTrain)
YPred = clfR.predict(XTest)
print ('YPred- Ridge CV')
print (YPred)
print (clfR.alpha_)

# Lasso Regression
clflml = linear_model.Lasso(alpha = 0.3)
clflml.fit(XTrain, YTrain)
YPred = clflml.predict(XTest)
print ('YPred- Lasso')
print (YPred)

# Bayesian Ridge
clflmlb = linear_model.BayesianRidge()
clflmlb.fit(XTrain, YTrain)
YPred = clflmlb.predict(XTest)
print ('YPred- Bayesian Ridge')
print (YPred)

# ARD Regression
clflmla = linear_model.ARDRegression(compute_score=True)
clflmla.fit(XTrain, YTrain)
YPred = clflmla.predict(XTest)
print ('YPred- ARD Regression')
print (YPred)


# Logistic Regression
LogReg = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
LogReg.fit(XTrain, YTrain)
# Finds the optimal model parameters using a least squares method.
# To get the parameter values:
LogReg.get_params()
# To predict a new input XTest,
YPred = LogReg.predict(XTest)
print ('YPred- Logistic Regression')
print (YPred)

# Support Vector Machines (SVM)
from sklearn import svm
clfs = svm.SVC()
clfs.fit(XTrain, YTrain)
# To predict a new input XTest
YPred = clfs.predict(XTest)
print ('YPred- Support Vector Machines')
print (YPred)


# Multi-class classification - Linear SVM
lin_clf = svm.LinearSVC()
lin_clf.fit(XTrain, YTrain)
# To predict a new input XTest
YPred = lin_clf.predict(XTest)
print ('YPred- Linear SVM')
print (YPred)

# Non Linear SVM
clfnl = svm.NuSVC()
clfnl.fit(XTrain, YTrain)
# To predict a new input XTest
YPred = clfnl.predict(XTest)
print ('YPred-  Non Linear SVM')
print (YPred)


from sklearn import tree
clft = tree.DecisionTreeClassifier()
clft = clft.fit(XTrain, YTrain)
YPred = clft.predict(XTest)
print ('YPred- DecisionTree')
print (YPred)

from sklearn.ensemble import RandomForestClassifier
clfrf = RandomForestClassifier(n_estimators=10)
clfrf = clfrf.fit(XTrain, YTrain)
YPred = clfrf.predict(XTest)
print ('YPred- RandomForest')
print (YPred)

importances = clfrf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clfrf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
for f in range(XTrain.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# 1. feature 3 (0.049330)
# 2. feature 28 (0.044043)
# 3. feature 1 (0.043618)
# 4. feature 0 (0.040636)
# 5. feature 10 (0.036512)
# 6. feature 41 (0.026741)
# 7. feature 21 (0.026472)
# 8. feature 39 (0.026281)
# 9. feature 16 (0.025645)
# 10. feature 34 (0.025267)
# 11. feature 24 (0.025020)
# 12. feature 6 (0.024642)
# 13. feature 25 (0.024294)
# 14. feature 8 (0.024108)
# 15. feature 17 (0.023959)
# 16. feature 7 (0.023713)
# 17. feature 11 (0.023668)
# 18. feature 43 (0.023294)
# 19. feature 23 (0.021770)
# 20. feature 19 (0.020901)
# 21. feature 14 (0.020533)
# 22. feature 4 (0.020321)
# 23. feature 35 (0.020292)
# 24. feature 12 (0.020274)
# 25. feature 9 (0.018432)
# 26. feature 37 (0.018308)
# 27. feature 29 (0.017650)
# 28. feature 5 (0.017550)
# 29. feature 45 (0.017185)
# 30. feature 22 (0.016746)
# 31. feature 46 (0.016625)
# 32. feature 30 (0.016298)
# 33. feature 20 (0.016200)
# 34. feature 18 (0.015977)
# 35. feature 27 (0.015920)
# 36. feature 38 (0.015346)
# 37. feature 15 (0.015253)
# 38. feature 13 (0.015172)
# 39. feature 44 (0.015146)
# 40. feature 40 (0.013770)
# 41. feature 2 (0.012762)
# 42. feature 26 (0.012662)
# 43. feature 42 (0.012180)
# 44. feature 31 (0.011869)
# 45. feature 36 (0.011366)
# 46. feature 32 (0.009596)
# 47. feature 33 (0.006655)
# 48. feature 47 (0.000000)


##Best is linear regression with one hot encoding of features,normalization of data.




