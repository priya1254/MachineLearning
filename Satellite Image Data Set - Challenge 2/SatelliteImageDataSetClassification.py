import pandas as pd
import numpy as np

df = pd.read_csv('C:\\Users\\priyagoe\\Downloads\\Anomaly-Detection-practical-master\\Anomaly-Detection-practical-master\\Challenges\\Satellite Image Data Set - Challenge 2\\sat_noisy.trn',',',header = None)
dftest = pd.read_csv('C:\\Users\\priyagoe\\Downloads\\Anomaly-Detection-practical-master\\Anomaly-Detection-practical-master\\Challenges\\Satellite Image Data Set - Challenge 2\\sat-test.csv.dat',',',header = None)
train_input = df.loc[:, 0:35]
train_labels = df.loc[:,36]
print("train input")
print(train_input)
print("train  labels")
print(train_labels)
print(dftest)
# ##rowmean
print("row mean")
rowlist = []
train_input_rowmean = train_input
# print(type(train_input_rowmean))
# print(type(train_input))
rowmean = train_input_rowmean.mean(axis = 1)
print(rowmean)
for iter, row in train_input_rowmean.iterrows():
    row = row.fillna(rowmean[iter])
    rowlist.append(row)
train_inputRowmean = pd.DataFrame(rowlist)
print(train_inputRowmean)
print("col mean")

##colmean
train_input_colmean = train_input
colmean = train_input_colmean.mean(axis = 0)
print(colmean)
train_input_colmean = train_input_colmean.fillna(colmean)
print(train_input_colmean)
##rowmode
##colmode
##rowmedian
print("row median")
train_input_rowmedian = train_input
rowlist = []
rowmedian = train_input_rowmedian.median(axis = 1)
print(rowmedian)
for iter, row in train_input_rowmedian.iterrows():
    row = row.fillna(rowmedian[iter])
    rowlist.append(row)
train_inputRowmedian = pd.DataFrame(rowlist)
print(train_inputRowmedian)
print("col median")
##colmeadian
train_input_colmedian = train_input
colmedian = train_input_colmedian.median(axis = 0)
print(colmedian)
train_input_colmedian = train_input_colmedian.fillna(colmedian)
print(train_input_colmedian)
print("row min")
##rowmin
train_input_rowmin = train_input
rowmin = train_input_rowmin.min(axis = 1)
rowlist = []
print(rowmin)
for iter, row in train_input_rowmin.iterrows():
    row = row.fillna(rowmin[iter])
    rowlist.append(row)
train_inputRowmin = pd.DataFrame(rowlist)
print(train_inputRowmin)
print("col min")
##colmin
train_input_colmin = train_input
colmin = train_input_colmin.min(axis = 0)
print(colmin)
train_input_colmin = train_input_colmin.fillna(colmin)
print(train_input_colmin)
print("row max")
##rowmax
train_input_rowmax = train_input
rowmax = train_input_rowmax.max(axis = 1)
rowlist = []
print(rowmax)
for iter, row in train_input_rowmax.iterrows():
    row = row.fillna(rowmax[iter])
    rowlist.append(row)
train_inputRowmax = pd.DataFrame(rowlist)
print(train_inputRowmax)
##colmax
print("col max")
train_input_colmax = train_input
colmax = train_input_colmax.max(axis = 0)
print(colmax)
train_input_colmax = train_input_colmax.fillna(colmax)
print(train_input_colmax)

# dfs = np.split(train_input, [9], axis=1)
# ##spectralrowmean
# for iter, row in train_input_rowmax.iterrows():
#     row = row.fillna(rowmax[iter])
#     rowlist.append(row)
# train_inputRowmax = pd.DataFrame(rowlist)
# print(train_inputRowmax)
##spectralrowmax
##spectralrowmin
##spectralrowmedian
#Multi Class SVM
from sklearn import svm
from sklearn.svm import NuSVC
def classifyMultiClassSVMClassifier(XTrain, XTest, YTrain, YTest):
    YPred = svm.SVC(kernel='linear').fit(XTrain, YTrain).predict(XTest)
    diff = YPred - YTest
    score = diff[diff == 0].size
    return (100.0 * score)/(YPred.size)
#One Vs Rest SVM Classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
def classifyOneVsRestClassifier(XTrain, XTest, YTrain, YTest):
    YPred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(XTrain, YTrain).predict(XTest)
    diff = YPred - YTest
    score = diff[diff == 0].size
    return (100.0 * score)/(YPred.size)
#One Vs One SVM Classifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
def classifyOneVsOneClassifier(XTrain, XTest, YTrain, YTest):
    YPred = OneVsOneClassifier(LinearSVC(random_state=0)).fit(XTrain, YTrain).predict(XTest)
    diff = YPred - YTest
    score = diff[diff == 0].size
    return (100.0 * score)/(YPred.size)
#Output Code SVM Classifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
def classifyOutputCodeClassifier(XTrain, XTest, YTrain, YTest):
    clf = OutputCodeClassifier(LinearSVC(random_state=0),code_size=2, random_state=0)
    YPred = clf.fit(XTrain, YTrain).predict(XTest)
    diff = YPred - YTest
    score = diff[diff == 0].size
    return (100.0 * score)/(YPred.size)
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
def classifyRandomForestClassifier(XTrain, XTest, YTrain, YTest):
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(XTrain, YTrain)
    YPred = clf.predict(XTest)
    diff = YPred - YTest
    score = diff[diff == 0].size
    return (100.0 * score)/(YPred.size)
#Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
def classifyGaussianNaiveBayesClassifier(XTrain, XTest, YTrain, YTest):
    gnb = GaussianNB()
    YPred = gnb.fit(XTrain, YTrain).predict(XTest)
    diff = YPred - YTest
    score = diff[diff == 0].size
    return (100.0 * score)/(YPred.size)
#Bernoulli Naive Bayes Classifier
from sklearn.naive_bayes import BernoulliNB
def classifyBernoulliNaiveBayesClassifier(XTrain, XTest, YTrain, YTest):
    bnb = BernoulliNB()
    YPred = bnb.fit(XTrain, YTrain).predict(XTest)
    diff = YPred - YTest
    score = diff[diff == 0].size
    return (100.0 * score)/(YPred.size)
#Multinomial Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
def classifyMultinomialNaiveBayesClassifier(XTrain, XTest, YTrain, YTest):
    mnb = MultinomialNB()
    YPred = mnb.fit(XTrain, YTrain).predict(XTest)
    diff = YPred - YTest
    score = diff[diff == 0].size
    return (100.0 * score)/(YPred.size)
#K Nearest Neighbours Classifier
from sklearn.neighbors import KNeighborsClassifier
def classifyKNNClassifier(XTrain, XTest, YTrain, YTest):
    neigh = KNeighborsClassifier(n_neighbors=3)
    YPred = neigh.fit(XTrain, YTrain).predict(XTest)
    diff = YPred - YTest
    score = diff[diff == 0].size
    return (100.0 * score)/(YPred.size)
#K Fold Cross Validation


from sklearn.model_selection import KFold
def kFoldCrossVal(XTrain, YTrain, classify):
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    score = 0.0
    XTrainorig = XTrain.as_matrix()
    YTrainorig = YTrain.as_matrix()
    print('XTrain:', XTrain)
    print('YTrain:', YTrain)
    for train_index, test_index in kf.split(XTrain):
            print('Train:', train_index)
            print('Test:', test_index)
            XTrainsplit = np.copy(XTrainorig)
            YTrainsplit = np.copy(YTrainorig)
            XTrain = XTrainsplit[train_index]
            XTest = XTrainsplit[test_index]
            YTrain = YTrainsplit[train_index]
            YTest = YTrainsplit[test_index]
            score1 = classify(XTrain, XTest, YTrain, YTest)
            print(score1)
            score+= score1
    return score/5


# from sklearn.model_selection import cross_val_score
# from sklearn import metrics
# def kFoldCrossVal(XTrain, YTrain, classify):
#      s = cross_val_score(classify, XTrain, YTrain, cv=5)
#      print("Accuracy: %0.2f (+/- %0.2f)" % (s.mean(), s.std() * 2))
#      return s.mean()
scoreOneVsRestClassifier = []
scoreOneVsOneClassifier = []
scoreOutputCodeClassifier = []
scoreRandomForestClassifier = []
scoreGaussianNaiveBayesClassifier = []
scoreBernoulliNaiveBayesClassifier = []
scoreMultinomialNaiveBayesClassifier = []
scoreMultiClassSVMClassifier = []
scoreKNNClassifier = []
def PreProcessStratifiedClassificationScores(XTrain, YTrain):
    score1 = kFoldCrossVal(XTrain, YTrain, classifyOneVsOneClassifier)
    print (score1)
    #score2 = kFoldCrossVal(XTrain, YTrain, classifyOutputCodeClassifier)
    #print score2
    score3 = kFoldCrossVal(XTrain, YTrain, classifyRandomForestClassifier)
    print(score3)
    score4 = kFoldCrossVal(XTrain, YTrain, classifyGaussianNaiveBayesClassifier)
    print(score4)
    score5 = kFoldCrossVal(XTrain, YTrain, classifyBernoulliNaiveBayesClassifier)
    print(score5)
    score6 = kFoldCrossVal(XTrain, YTrain, classifyMultinomialNaiveBayesClassifier)
    print(score6)
    score7 = kFoldCrossVal(XTrain, YTrain, classifyMultiClassSVMClassifier)
    print(score7)
    score8 = kFoldCrossVal(XTrain, YTrain, classifyKNNClassifier)
    print(score8)
    #     scoreOneVsRestClassifier.append(score)




    scoreOneVsOneClassifier.append(score1)
    #     scoreOutputCodeClassifier.append(score2)
    scoreRandomForestClassifier.append(score3)
    scoreGaussianNaiveBayesClassifier.append(score4)
    scoreBernoulliNaiveBayesClassifier.append(score5)
    scoreMultinomialNaiveBayesClassifier.append(score6)
    scoreMultiClassSVMClassifier.append(score7)
    scoreKNNClassifier.append(score8)


PreProcessStratifiedClassificationScores(train_inputRowmean,train_labels)
PreProcessStratifiedClassificationScores(train_input_colmean,train_labels)
PreProcessStratifiedClassificationScores(train_inputRowmedian,train_labels)
PreProcessStratifiedClassificationScores(train_input_colmedian,train_labels)
PreProcessStratifiedClassificationScores(train_inputRowmin,train_labels)
PreProcessStratifiedClassificationScores(train_input_colmin,train_labels)
PreProcessStratifiedClassificationScores(train_inputRowmax,train_labels)
PreProcessStratifiedClassificationScores(train_input_colmax,train_labels)
df1 = pd.DataFrame( scoreOneVsOneClassifier)
df1.to_csv(' scoreOneVsOneClassifier.csv')
df2 = pd.DataFrame( scoreRandomForestClassifier)
df1.to_csv(' scoreRandomForestClassifier.csv')
df1 = pd.DataFrame( scoreGaussianNaiveBayesClassifier)
df1.to_csv(' scoreGaussianNaiveBayesClassifier.csv')
df1 = pd.DataFrame( scoreBernoulliNaiveBayesClassifier)
df1.to_csv('scoreBernoulliNaiveBayesClassifier.csv')
df1 = pd.DataFrame( scoreMultinomialNaiveBayesClassifier)
df1.to_csv(' scoreMultinomialNaiveBayesClassifier.csv')
df1 = pd.DataFrame( scoreMultiClassSVMClassifier)
df1.to_csv('scoreMultiClassSVMClassifier.csv')
df1 = pd.DataFrame( scoreKNNClassifier)
df1.to_csv('scoreKNNClassifier.csv')

import matplotlib.pyplot as plt

#matplotlib inline
#plt.plot(scoreOneVsRestClassifier, label = "One Vs Rest Classifier")
scoreOneVsOneClassifier = pd.read_csv('C:\\Users\\priyagoe\\challenges\\ scoreOneVsOneClassifier.csv')
scoreRandomForestClassifier = pd.read_csv('C:\\Users\\priyagoe\\challenges\\ scoreRandomForestClassifier.csv')
scoreGaussianNaiveBayesClassifier = pd.read_csv('C:\\Users\\priyagoe\\challenges\\ scoreGaussianNaiveBayesClassifier.csv')
scoreBernoulliNaiveBayesClassifier = pd.read_csv('C:\\Users\\priyagoe\\challenges\\scoreBernoulliNaiveBayesClassifier.csv')
scoreMultinomialNaiveBayesClassifier = pd.read_csv('C:\\Users\\priyagoe\\challenges\\ scoreMultinomialNaiveBayesClassifier.csv')
scoreMultiClassSVMClassifier = pd.read_csv('C:\\Users\\priyagoe\\challenges\\scoreMultiClassSVMClassifier.csv')
scoreKNNClassifier = pd.read_csv('C:\\Users\\priyagoe\\challenges\\scoreKNNClassifier.csv')
plt.plot(scoreOneVsOneClassifier, label = "One Vs One Classifier")
#plt.plot(scoreOutputCodeClassifier, label = "Output Code Classifier")
plt.plot(scoreRandomForestClassifier, label = "Random Forest Classifier")
plt.plot(scoreGaussianNaiveBayesClassifier, label = "Gaussian Naive Bayes Classifier")
plt.plot(scoreBernoulliNaiveBayesClassifier, label = "Bernoulli Naive Bayes Classifier")
plt.plot(scoreMultinomialNaiveBayesClassifier, label = "Multinomial Naive Bayes Classifier")
plt.plot(scoreMultiClassSVMClassifier, label = "Multiclass Linear SVM Classifier")
plt.plot(scoreKNNClassifier, label = "KNN Classifier")


##Result Random Forest Classifier n = 50, gain = entropy with replacement of missing values = interpolate worked best, or rowpectral median values for missing data
# with rearranging pixels worked best among all.