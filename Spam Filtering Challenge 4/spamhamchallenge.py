import numpy as np
import pandas as pd
import os

os.chdir('C:\\Users\\priyagoe\\Downloads\\Anomaly-Detection-practical-master\\Anomaly-Detection-practical-master\\Challenges\\Spam Filtering Challenge 4\\')
XTrain = np.loadtxt('spam-mail.tr.label', skiprows=1, delimiter=',')
YTrain = XTrain[:,-1]
YTrain = YTrain.astype(np.int)
XTrainFileNames = XTrain[:,0]
XTrainFileNames = XTrainFileNames.astype(np.int)
XTestFileNames = XTrain[:1827,0]  ##which needs prediction
XTestFileNames = XTestFileNames.astype(np.int)
print (XTrainFileNames.shape,YTrain.shape, XTestFileNames.shape)
print (XTrainFileNames[:5],YTrain[:5])
trainFolder = os.getcwd()+os.sep+'TR-mails'+os.sep+'TR'+os.sep
testFolder = os.getcwd()+os.sep+'TT-mails'+os.sep+'TT'+os.sep
print (trainFolder,testFolder)

import email
with open(trainFolder+'TRAIN_'+str(XTrainFileNames[0])+'.eml','r') as f:
#     print f.readlines(),len(f.readlines())
    message = email.message_from_file(f)
#     headers = email.message_from_file(f, headersonly)
print (message)#,headers
print (type(message))


print (message.keys())
print (message.values())
print (message['Return-Path'], message['Message-Id'])
print (message['Received'])
print (message['Subject'])
print (message['From'])
print (message['Sender'])

parser = email.parser.HeaderParser()
headers = parser.parsestr(message.as_string())
for h in headers.items():
    print (h)
print (type(headers), dir(headers), len(headers))

def getEmailBody(messageStr):
    b = email.message_from_string(messageStr)
    strPayload = ''
    if b.is_multipart():
        for payload in b.get_payload():
            # if payload.is_multipart(): ...
            strPayload += str(payload.get_payload())
    else:
        strPayload += str(b.get_payload())
    print("strPayload is")
    print(strPayload)
    return strPayload

def readEmailBody(XTrainFileNames, folder, fileString='TRAIN_'):
    cnt = 0
    trainEmailBody = []
    for emailFileName in XTrainFileNames:
            f = open(folder+fileString+str(emailFileName)+'.eml','rb')
            #message = f.readlines()
            #message = message.tostring
            if(emailFileName == 70):
                break
            cnt+=1
            print(cnt)
            message = email.message_from_file(f)
            print("msg as string is")
            print(message.as_string())
            trainEmailBody.append(getEmailBody(message.as_string()))
            print(trainEmailBody[cnt-1])
            f.close()
    return trainEmailBody

trainEmailBody = readEmailBody(XTrainFileNames, trainFolder, 'TRAIN_')
print (trainEmailBody[0])
print (len(trainEmailBody))

testEmailBody = readEmailBody(XTestFileNames, testFolder, 'TEST_')
print (testEmailBody[0])
print (len(testEmailBody))

from collections import Counter
def AnalyseWordLabelContribution(XMessages, YTrain):
    XFeaturesSpam = []
    XFeaturesHam = []
    i = 0
    for message in XMessages:
        if message == None:
            continue
        else:
            wordList = message.split()
            for word in wordList:
                if(YTrain[i] == 0):
                    XFeaturesSpam.append((word.lower(), YTrain[i]))
                elif (YTrain[i] == 1):
                    XFeaturesHam.append((word.lower(), YTrain[i]))

            i+=1
    return Counter(XFeaturesSpam), Counter(XFeaturesHam)

def computeSpamicityWords(XFeaturesSpam, XFeaturesHam):
    XFeatures = {}
    for key in XFeaturesSpam.keys():
        word = key[0]
    #     print word, XFeaturesSpam[key], XFeaturesHam[(word, 1)]
        spamicity = float(XFeaturesSpam[key]) / float(XFeaturesSpam[key] + XFeaturesHam[(word, 1)])
#         hamicity = 1-spamicity
        XFeatures[word] = spamicity
    return XFeatures

def extractEmailSubjects(folder,fileString, emailFileName):
    with open(folder+fileString+str(emailFileName)+'.eml','r') as f:
        message = email.message_from_file(f)
    return message['Subject']

def readSubjects(XTrainFileNames, trainFolder, fileString='TRAIN_'):
    trainSubjects = []
    for fileName in XTrainFileNames:
        if (fileName == 70):
            break
        trainSubjects.append(extractEmailSubjects(trainFolder,fileString,fileName))
    return trainSubjects



trainSubjects = readSubjects(XTrainFileNames, trainFolder, 'TRAIN_')
print(len(trainSubjects))
XFeaturesSubjectSpam, XFeaturesSubjectHam = AnalyseWordLabelContribution(trainSubjects, YTrain)
# print XFeaturesSpam
##spamicity of each word belonging to list of spam words
SubjectWordsSpamicity = computeSpamicityWords(XFeaturesSubjectSpam, XFeaturesSubjectHam)

def computeSpamHamMetrics(message, wordSpamicity):
    spamInd =1.0
    hamInd =1.0
    if message == None:
        return 0.5,0.5
    else:
        wordList = message.split()
        for word in wordList:
            try:
                spamInd *= wordSpamicity[word.lower()]
                hamInd *= (1-wordSpamicity[word.lower()])
            except KeyError:
                spamInd *= 1.0#0.5
                hamInd *= 1.0#0.5
    return spamInd,hamInd


##consider words with top 15 spamicity in a message to cotribute to spam,ham,metric
def computeTopSpamHamMetrics(message, wordSpamicity, noOfWords=15):
    spamInd = 1.0
    hamInd = 1.0
    spamList = []
    hamList = []
    if message == None:
        return 0.5, 0.5
    else:
        wordList = message.split()
        for word in wordList:
            try:
                spamList.append(abs(wordSpamicity[word.lower()] - 0.5))
                hamList.append(abs(wordSpamicity[word.lower()] - 1.0))
            except KeyError:
                spamList.append(0.5)
                hamList.append(1.0)
        spamList.sort(reverse=True)
        hamList.sort(reverse=True)

    if len(spamList) > noOfWords:
        for i in range(noOfWords):
            spamInd *= spamList[i]
            hamInd *= hamList[i]
    else:
        for i in range(len(spamList)):
            spamInd *= spamList[i]
            hamInd *= hamList[i]

    return spamInd, hamInd


from sklearn.cross_validation import StratifiedKFold

XTrainFileNames = XTrain[:69,0]
def stratifiedKFoldVal(XTrain1, YTrain1, classify):
    n_folds = 5
    score = 0.0
    skf = StratifiedKFold(YTrain1, n_folds)
    trainSubjects = readSubjects(XTrainFileNames.astype(np.int), trainFolder, 'TRAIN_')



    for train_index, test_index in skf:
        X_train, X_test = XTrain1[train_index], XTrain1[test_index]
        y_train, y_test = YTrain1[train_index], YTrain1[test_index]

        x_test_subs = [trainSubjects[i] for i in test_index]
        x_train_subs = [trainSubjects[i] for i in train_index]
        score += classify(X_train, X_test, y_train, y_test)
        print('score')
        print(score)
    return score / n_folds


# Naive Bayes Classifier
##http://airccse.org/journal/jcsit/0211ijcsit12.pdf
def classifySpamHam(XTrain, XTest, YTrain, YTest):
    threshold = 0.5
    XTrainSubjects = readSubjects(XTrain.astype(np.int), trainFolder,  'TRAIN_')
    print('XTrainSubjects')
    print(XTrainSubjects)
    XFeaturesSubjectSpam, XFeaturesSubjectHam = AnalyseWordLabelContribution(XTrainSubjects, YTrain)
    SubjectWordsSpamicity = computeSpamicityWords(XFeaturesSubjectSpam, XFeaturesSubjectHam)

    XTestSubjects = readSubjects(XTest.astype(np.int), trainFolder,  'TRAIN_')
    print('XTestSubjects')
    print(XTestSubjects)
    YPred = []
    for subject in XTestSubjects:
        spam, ham = computeSpamHamMetrics(subject, SubjectWordsSpamicity)
        print('spam, ham')
        print(spam, ham)
        spamminess = (1 + spam - ham) / 2
        print('spammines')
        print(spamminess)
        if spamminess > threshold:
            YPred.append(0)
        else:
            YPred.append(1)

    YPred = np.array(YPred)
    print('YPred')
    print(YPred)
    print('YTest')
    print(YTest)
    diff = YPred - YTest
    score = diff[diff == 0].size / float(YPred.size)
    print('score')
    print(score)
    return (100.0 * score)



XTrain1 = XTrain[:69,0]
YTrain1 = YTrain[:69]
stratifiedKFoldVal(XTrain1, YTrain1, classifySpamHam)

def classifySpamHamTop(XTrain, XTest, YTrain, YTest):
    threshold = 0.5
    XTrainSubjects = readSubjects(XTrain.astype(np.int), trainFolder,  'TRAIN_')
    print('XTrainSubjects')
    print(XTrainSubjects)
    XFeaturesSubjectSpam, XFeaturesSubjectHam = AnalyseWordLabelContribution(XTrainSubjects, YTrain)
    SubjectWordsSpamicity = computeSpamicityWords(XFeaturesSubjectSpam, XFeaturesSubjectHam)

    XTestSubjects = readSubjects(XTest.astype(np.int), trainFolder,  'TRAIN_')
    print('XTestSubjects')
    print(XTestSubjects)
    YPred = []
    for subject in XTestSubjects:
        spam, ham = computeTopSpamHamMetrics(subject, SubjectWordsSpamicity)
        print('spam, ham')
        print(spam, ham)
        spamminess = (1 + spam - ham) / 2
        print('spammines')
        print(spamminess)
        if spamminess > threshold:
            YPred.append(0)
        else:
            YPred.append(1)

    YPred = np.array(YPred)
    print('YPred')
    print(YPred)
    print('YTest')
    print(YTest)
    diff = YPred - YTest
    score = diff[diff == 0].size / float(YPred.size)
    print('score')
    print(score)
    return (100.0 * score)



stratifiedKFoldVal(XTrain1, YTrain1, classifySpamHamTop)

##Random Forest with [spam,ham] of  a doc as feature of a doc
from sklearn.ensemble import RandomForestClassifier
# Create feature vector using computeSpamHamMetrics()
def createSpamHamFeature(messages, wordSpamicity):
    XFeatures = []
    for message in messages:
        spam, ham = computeSpamHamMetrics(message, wordSpamicity)
        XFeatures.append([spam,ham])
    return np.array(XFeatures)

def createTopSpamHamFeature(messages, wordSpamicity):
    XFeatures = []
    for message in messages:
        spam, ham = computeTopSpamHamMetrics(message, wordSpamicity)
        XFeatures.append([spam,ham])
    return np.array(XFeatures)

def classifySpamHamRF(XTrain, XTest, YTrain, YTest):
    XTrainSubjects = readSubjects(XTrain.astype(np.int), trainFolder, 'TRAIN_')
    print('XTrainSubjects')
    print(XTrainSubjects)
    XFeaturesSubjectSpam, XFeaturesSubjectHam = AnalyseWordLabelContribution(XTrainSubjects, YTrain)
    SubjectWordsSpamicity = computeSpamicityWords(XFeaturesSubjectSpam, XFeaturesSubjectHam)

    XTestSubjects = readSubjects(XTest.astype(np.int), trainFolder, 'TRAIN_')
    print('XTestSubjects')
    print(XTestSubjects)
    XTrainSubjectFeatures = createSpamHamFeature(XTrainSubjects, SubjectWordsSpamicity)
    XTestSubjectFeatures = createSpamHamFeature(XTestSubjects, SubjectWordsSpamicity)
    print('XTrainSubjectFeatures')
    print(XTrainSubjectFeatures)
    print('XTestSubjectFeatures')
    print(XTestSubjectFeatures)
    clf = RandomForestClassifier(n_estimators=200,criterion='entropy')
    clf.fit(XTrainSubjectFeatures, YTrain)
    YPred = clf.predict(XTestSubjectFeatures)
    print('YPred')
    print(YPred)
    print('YTest')
    print(YTest)
    diff = YPred - YTest
    score = diff[diff == 0].size / float(YPred.size)
    print('score')
    print(score)
    return (100.0 * score)

stratifiedKFoldVal(XTrain1, YTrain1, classifySpamHamRF)


##
# KNeighborsClassifier with [spam,ham] of  a doc as feature of a doc
from sklearn.neighbors import KNeighborsClassifier

def classifySpamHamKNN(XTrain, XTest, YTrain, YTest):
    XTrainSubjects = readSubjects(XTrain.astype(np.int), trainFolder, 'TRAIN_')
    print('XTrainSubjects')
    print(XTrainSubjects)
    XFeaturesSubjectSpam, XFeaturesSubjectHam = AnalyseWordLabelContribution(XTrainSubjects, YTrain)
    SubjectWordsSpamicity = computeSpamicityWords(XFeaturesSubjectSpam, XFeaturesSubjectHam)

    XTestSubjects = readSubjects(XTest.astype(np.int), trainFolder, 'TRAIN_')
    print('XTestSubjects')
    print(XTestSubjects)
    XTrainSubjectFeatures = createSpamHamFeature(XTrainSubjects, SubjectWordsSpamicity)
    XTestSubjectFeatures = createSpamHamFeature(XTestSubjects, SubjectWordsSpamicity)
    print('XTrainSubjectFeatures')
    print(XTrainSubjectFeatures)
    print('XTestSubjectFeatures')
    print(XTestSubjectFeatures)
    neigh = KNeighborsClassifier(n_neighbors=21)
    YPred = neigh.fit(XTrainSubjectFeatures, YTrain).predict(XTestSubjectFeatures)
    print('YPred')
    print(YPred)
    print('YTest')
    print(YTest)
    diff = YPred - YTest
    score = diff[diff == 0].size / float(YPred.size)
    print('score')
    print(score)
    return (100.0 * score)

stratifiedKFoldVal(XTrain1, YTrain1, classifySpamHamKNN)


XFeaturesBodySpam, XFeaturesBodyHam = AnalyseWordLabelContribution(trainEmailBody, YTrain)
BodyWordsSpamicity = computeSpamicityWords(XFeaturesBodySpam, XFeaturesBodyHam)
print (len(BodyWordsSpamicity))
# print BodyWordsSpamicity

def extractEmailField(folder, fileString, emailFileName, fieldName):
    with open(folder+fileString+emailFileName+'.eml','r') as f:
        message = email.message_from_file(f)
    return message[fieldName]

def readEmailField(fieldName, XTrainFileNames, trainFolder, fileString='TRAIN_'):
    trainField = []
    for fileName in XTrainFileNames:
        trainField.append(extractEmailField(trainFolder,fileString,fileName,fieldName))
    return trainField


def extractEmail(folder,fileString, emailFileName):
    with open(folder+fileString+emailFileName+'.eml','r') as f:
        message = email.message_from_file(f)
    return message.as_string()

def readEmails(XTrainFileNames, trainFolder, fileString='TRAIN_'):
    trainEmails = []
    for fileName in XTrainFileNames:
        trainEmails.append(extractEmail(trainFolder,fileString,fileName))
    return trainEmails

#fieldName = 'Sender'
fieldName = 'From'
trainSender = readEmailField(fieldName, XTrainFileNames, trainFolder, 'TRAIN_')
testSender = readEmailField(fieldName, XTestFileNames, testFolder, 'TEST_')

fieldName = 'Date'
trainDate = readEmailField(fieldName, XTrainFileNames, trainFolder, 'TRAIN_')
testDate= readEmailField(fieldName, XTestFileNames, testFolder, 'TEST_')


import datetime

minYr = 2000
maxYr = 2010


def CheckFutureDate(lstYear):
    futureYr = []
    for yr in lstYear:
        if yr >= minYr and yr <= maxYr:
            futureYr.append(True)
        else:
            if yr == 2021:
                futureYr.append(True)
            else:
                #             print emailDate
                futureYr.append(False)
    return futureYr


def getDates(lstDates):
    dateList = []
    # fmt = "%a, %d %b %Y %H:%M:%S"
    # date_object = datetime.strptime(string_date[:25], fmt)
    # CheckFutureDate(date_object)
    monthList = ['Jan ', 'Feb ', 'Mar ', 'Apr ', 'May ', 'Jun ', 'Jul ', 'Aug ', 'Sep ', 'Oct ', 'Nov ', 'Dec ']
    for string_date in lstDates:
        for i in monthList:
            strYearPart  = string_date.split(i)
            if len(strYearPart) == 2:
                year = strYearPart[1][:4]
                try:
                    year = int(year)
                except:
                    year = int('20' + year.split()[0])
    #                 print year, string_date

#                 dateList.append(CheckFutureDate(year))
                dateList.append(year)

    return dateList

lstTrainYear = getDates(trainDate)
lstTestYear = getDates(testDate)


def CheckMissingSender(lstSender):
    missingSenders = []
    for sender in lstSender:
        if sender is None:
            missingSenders.append(True)
        else:
            missingSenders.append(False)
    return missingSenders

##vowpal wabbit works the best coz it can work on whole raw email data, email subjects + email body + other data.