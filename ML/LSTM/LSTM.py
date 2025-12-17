import numpy as np
import pandas as pd
import keras as ke
import tensorflow as tf;
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import datetime
import time


import matplotlib.pyplot as plt
import tensorflow as tf
path = r'../Data1/NationalNames.csv'
data = pd.read_csv(path)
#'Id' 'Name' 'Year' 'Gender' 'Count'
#print(data['Id'])
#data['Name'] = data['Name']

data2 = np.array(data).reshape(-1,5)
print(data2[0])
data3 = data2[data2[:,1]=='Kevin']
#print(data3[data3[:,4]==max(data3[:,4])])
data4 = np.linspace(start=10,stop=100,num=10);
inp = np.zeros((7,3));
for i in range(7):
    inp[i] = data4[i:i+3];
y = data4[3:10]
print(inp)
print(y)

def split_sequence(sequence,n_steps):
    X,y = list(),list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix >= len(sequence)-1:
            break
        seq_x,seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
n_steps = 3
n_features = 1
x1,y1 = split_sequence(data4,n_steps)
print(x1)
print(y1)
print(x1)
print(y1)
model = tf.keras.Sequential()
model.add(LSTM(units=60,activation='relu',input_shape=(n_steps,n_features)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')
n_features = 1
X = x1.reshape((x1.shape[0], x1.shape[1], n_features))
#model.fit(X, y1, epochs=400, verbose=0)
print(X)
# demonstrate prediction
x_input = np.array([100, 200, 300])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)


path1 = r'Data1/XAUUSD@60.csv'
stockData60 = pd.read_csv(path1)
print(stockData60)
STGold60 = np.array(stockData60).reshape(-1,7)
print(STGold60[0])
print(STGold60[0][0]) #Date

date_format = '%Y.%m.%d'
d = datetime.datetime.strptime(STGold60[0][0],date_format)
ppp = int(d.strftime("%Y%m%d%H%M%S"))
print("Date Int : " , ppp) #Date
print(STGold60[0][1]) #Time
print(STGold60[0][2]) #Open
print(STGold60[0][3]) #High
print(STGold60[0][4]) #Low
print(STGold60[0][5]) #Close
print(STGold60[0][6]) #Amount
