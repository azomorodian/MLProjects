import yfinance as yf
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import sklearn

from sklearn.preprocessing import MinMaxScaler
import datetime
import warnings
warnings.filterwarnings("ignore")
#step 1 Download finance data
#choose a stack ticker symbole, 'AAPL' for Apple data
ticker = 'AAPL'
#Download Historical Data from yahoo finance
data = yf.download(ticker,start= '2020-01-01',end=datetime.datetime.today().strftime('%Y-%m-%d'))
#focuse on 'Close' prices for simplicity
print(data.to_string())
t_len = int(np.ceil(len(data)*0.8))
print("Last Train Data")
print(data.iloc[t_len])
print(t_len)
print(len(data))
print(data.iloc[len(data)-1])
print(data.iloc[-10:])
exit()


data = data[['Close']]
print(data.to_string())
plt.plot(data)
plt.show()



#step2 Preprocess data
#initilize MinMaxScaler to normalize the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))

#scale data for training
scaled_data = scaler.fit_transform(data)
#print(scaled_data)

#Step 3: Prepare Trainning Data
#define training data length as 80% of total data
trsining_data_len = int(np.ceil(len(scaled_data)*0.8))




#print(data[trsining_data_len])
#Split the scaled data into training set
train_data = scaled_data[0:int(trsining_data_len),:]
print(train_data)

#create empty lists for features (x_train and target (y_train))
x_train = []
y_train = []

#population x_train with 60 days of data and y_train 30 in the following day's closing price
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])   #Past 60 days
    y_train.append(train_data[i,0])        #Target for the next day's close price
#convert List to numpy array for model training
x_train , y_train = np.array(x_train) , np.array(y_train)

#reshape x_train to format [samples, time steps, features] required for the LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#Step 4: Build LSTM Model
model = Sequential()

#First LSTM Layer with 50 units and return the sequences
model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2)) #Dropout layers to prevent over fitting
#Second Layer : LSTM Layer
model.add(LSTM(units=50,return_sequences=False))
model.add(Dropout(0.2))  #Dropout layers to prevent over fitting
#ِDense Layet 25 units
model.add(Dense(units=25))
#Output layer with only one unit
model.add(Dense(units=1))

#Compile the model using "ADAM" optimizer and mean squre error as the loss function
model.compile(optimizer='adam',loss='mean_squared_error')


#Step5: Train the model
#Train the model with batch size of 1 and 1 epoch (adjust epoch count for better results)
model.fit(x_train,y_train,batch_size=1,epochs=1)

#Step 6: prepare the data for 30 days forcast
#take the last 60 days from the dataset for generating future predictions
last_60_days = scaled_data[-60:]
x_future = last_60_days.reshape((1,last_60_days.shape[0],1))
print(x_future.shape)
print(x_future)


#Step 7: Generate 30 days forecast
#Create an empty list to store prediction for next 30 days
future_prediction = []
for _ in range(30):
    pred = model.predict(x_future)
    future_prediction.append(pred[0,0])
    x_future = np.append(x_future[:,1:,:], [pred], axis=1)
    print("X Future : ",x_future.shape)
    print(x_future)
    print("pred : ", pred)


#x_future = np.append(x_future,[[future_prediction]],axis=1)
#
#print(x_future)
#Step 8: Transfrom prediction  back to the original scale
#Convert the scaled predictions back to the original scale using inverse_Transform
future_prediction = scaler.inverse_transform(np.array(future_prediction).reshape(-1,1))

#Step 9 : Visualize the results
#create a Dataframe to hold the 30-day forcast with dates
#print(future_prediction)
forecast_dates = pd.date_range(start=data.index[-1]+pd.Timedelta(days=1), periods=30,freq='D')
forecast = pd.DataFrame(future_prediction,index=forecast_dates,columns=['prediction'])

#Plot hjistorical data and future predictions for comparison
plt.figure(figsize=(10,5))
plt.plot(data['Close'],label='Historical prices')
plt.plot(forecast, label='30 days forecast',linestyle='dotted',color='Orange')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()





