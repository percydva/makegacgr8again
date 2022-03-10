import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forex_python.converter import CurrencyRates
import seaborn as sns

def read_file(filename1, filename2):
    ads_data = pd.read_csv(filename1, parse_dates=[0])
    nke_data = pd.read_csv(filename2, parse_dates=[0])
    return ads_data, nke_data

def currency_rates(data):
    c = CurrencyRates()
    data['exchange_rate'] = data.apply(lambda x: c.get_rate('EUR', 'USD', x.Date), axis=1)
    return data

def exchange_currency(data):
    data['Open'] = data.apply(lambda x: x.Open*x.exchange_rate, axis=1)
    data['Close'] = data.apply(lambda x: x.Close*x.exchange_rate, axis=1)
    data['Adj Close'] = data.apply(lambda x: x['Adj Close']*x.exchange_rate, axis=1)
    return data

def plot_price_history(data, s):
    sns.set_theme()
    sns.set_style('darkgrid')
    plt.figure(figsize=(16,8))
    plt.plot(data['Date'], data['Close'], label='Close Price History', linewidth=4)
    plt.plot(data['Date'], data['Open'], label='Open Price History')
    plt.plot(data['Date'] ,data['Adj Close'], label='Adj Close Price History')
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Price History in USD')
    plt.legend()
    plt.xticks(rotation=65)
    plt.title(s+' Price History from March 2021 to March 2022')
    plt.savefig('./output/'+s+'_price_history.png')
    plt.show()

def take_max(data):
    return data.loc[data['Close'].idxmax()]

def split_data(stock, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    # create all possible sequences of length seq_len
    for index in range(len(data_raw)-lookback): 
        data.append(data_raw[index:index+lookback])
    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]
    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]
    return [x_train, y_train, x_test, y_test]


def main(filename1, filename2):
    print('This should take roughly 5 minutes, please wait!')
    ads_data, nke_data = read_file(filename1, filename2)
    ads_data = currency_rates(ads_data)
    ads_data = exchange_currency(ads_data)
    if len(filename1) < 20:
        output1 = 'ads'
        output2 = 'nke'  
    else:
        output1 = 'ads_monthly'
        output2 = 'nke_monthly'
    ads_data.to_csv('./data/'+output1+'_preprocessing.csv')
    plot_price_history(ads_data, output1)
    plot_price_history(nke_data, output2)
    print('ADS:\n', take_max(ads_data))
    print('NKE:\n', take_max(nke_data))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Example command: python3 data_preprocessing.py ./data/<filename1>.csv ./data/<filename2>.csv')
    else:
        main(sys.argv[1], sys.argv[2])