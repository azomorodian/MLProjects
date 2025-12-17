import math
import random

import numpy as np
import pandas as pd
from jedi.api.refactoring import inline
import matplotlib.pyplot as plt



def encode_class(mydata):
    classes = []
    for i in range (len(mydata)):
        if mydata.iloc[i,-1] not in  classes:
            classes.append(mydata.iloc[i,-1])
    for i in range(len(mydata)):
        for j in range(len(classes)):
            if mydata.iloc[i,-1] == classes[j]:
                mydata.iloc[i,-1] = j
    return mydata
def splitting(mydata,ratio):
    train_num = int(len(mydata)*ratio)
    train = []
    test = mydata.values.tolist()

    while len(train) < train_num:
        index = random.randrange(len(test))
        train.append(test.pop(index))
    return train,test
def groupUnderClass(mydata):
    data_dict = {}
    for i in range (len(mydata)):
        if mydata[i][-1] not in data_dict:
            data_dict[mydata[i][-1]] = []
        data_dict[mydata[i][-1]].append(mydata[i])
    return data_dict
def MeanAndStdDev(numbers):
    avg = np.mean(numbers)
    stddv = np.std(numbers)
    return avg,stddv
def MeanAndStdDevForClass(mydata):
    info = {}
    data_dict = groupUnderClass(mydata)
    for classValue, instances in data_dict.items():
        info[classValue] = [MeanAndStdDev(attribute) for attribute in zip(*instances)]
    return info
def CalculateGaussianProbability(x,mean,stdev):
    epsilon = 1e-10
    expo = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev+epsilon,2))))
    return (1/(math.sqrt(2*math.pi)*(stdev + epsilon)))*expo
def CalculateClassProbabilities(info,test):
    probabilities = {}
    for classValue, classSummeries in info.items():
        probabilities[classValue] = 1
        for i in range(len(classSummeries)):
            mean , stdev = classSummeries[i]
            x = test[i]
            probabilities[classValue] *= CalculateGaussianProbability(x,mean,stdev)
    return probabilities
def predict(info, test):
    probabilities = CalculateClassProbabilities(info, test)
    bestLabel = max(probabilities, key=probabilities.get)
    return bestLabel

def getPredictions(info, test):
    predictions = [predict(info, instance) for instance in test]
    return predictions

def accuracy_rate(test, predictions):
    correct = sum(1 for i in range(len(test)) if test[i][-1] == predictions[i])
    return (correct / float(len(test))) * 100.0

df = pd.read_csv('../input/diabete/diabetes.csv')
pd.set_option('display.max_rows', None)
print(df.head())
print(df.columns)
print(df.Outcome.unique())
print(len(df))

#mydata = encode_class(df)
#mydata = encode_class(df)
mydata = df.copy()
print(mydata.Outcome.unique())
cols = [col for col in mydata.columns if col != 'Outcome']
mydata[cols] = mydata[cols].apply(pd.to_numeric)
mydata['Outcome'] = mydata['Outcome'].astype(int)


ratio = 0.711111
train_data, test_data = splitting(mydata, ratio)
print(train_data)
print(test_data)
print('Total number of examples:', len(mydata))
print('Training examples:', len(train_data))
print('Test examples:', len(test_data))
t = train_data[3][-1]
print(t)
info = MeanAndStdDevForClass(train_data)
predictions = getPredictions(info, test_data)
print(predictions)
accuracy = accuracy_rate(test_data, predictions)
print('Accuracy of the model:', accuracy)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_true = [row[-1] for row in test_data]
y_pred = predictions

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()
print(np.shape(train_data))


