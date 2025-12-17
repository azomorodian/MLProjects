import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import numpy as np

def str_to_datetime(DateStr,TimeStr):
    spDate = DateStr.split('.')
    spTime = TimeStr.split(':')
    year, month, day = int(spDate[0]), int(spDate[1]), int(spDate[2])
    hour, minute = int(spTime[0]), int(spTime[1])
    return datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute)

def df_to_windowed_df(dataframe,first_date_str,last_date_str,n=3):
    first_date = str_to_datetime(first_date_str,"10:00")
    last_date = str_to_datetime(last_date_str,"23:00")

    target_date = last_date

    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n + 1)

        if len(df_subset) != n + 1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset[5].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

        if last_time:
            break

        target_date = next_date

        if target_date == last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)

    for i in range(0, n):
        X[:, i]
        ret_df[f'Target-{n - i}'] = X[:, i]

    ret_df['Target'] = Y

    return ret_df


headerList = ['Date', 'Time', 'Open', 'High', 'Low', 'Close','Volume']
df = pd.read_csv('../Data1/XAUUSD@60.csv', names=headerList)

#Date Time open High Low Close Volume
#print(df.to_string())

df['Date'] = df[['Date','Time']].apply(lambda x: str_to_datetime(*x),axis=1)
df.drop(columns=['Time'],inplace=True)

df2 = df[['Date','Close']]

df2.set_index('Date',inplace=True)

#print(df2.to_string())

#print(df)
#print(df[[0,1,5]])

#
#df = df[[1,5]]
#df.index = df.pop(1)
#x = np.linspace(1,10,10)
#y = np.linspace(1,10,10)

#plt.plot(df2.Close)
#plt.show()

#df.rename(columns={'[5]':'Close'},inplace=True)

#print("Coloumns")
#print(df.columns)

#windowed_df = df_to_windowed_df(df2,
#                                '2024.08.05',
#                                '2025.01.17',
#                                n=3)
#print("Windowed Dataframe:")
#print(windowed_df)


#subdata = df.loc[:datetime.datetime(year=2024,month=8,day=5,hour=13,minute=0)].tail(4)
#print("Sub Dataframe:")
#print(subdata)
#
dt = str_to_datetime('2025.01.17','23:00')
print(df2.loc[:dt].tail(4).to_string())


df_subset = df2.loc[:dt].tail(4)

values = df_subset['Close'].to_numpy()
print("************ Values **************")
print(values)

x, y = values[:-1], values[-1]
print("************ X **************")
print(x)
print("************ y **************")
print(y)
