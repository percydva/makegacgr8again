from dataclasses import dataclass
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forex_python.converter import CurrencyRates
import seaborn as sns

def read_file(filename):
    return pd.read_csv(filename, parse_dates=[0])

def currency_rates(data):
    c = CurrencyRates()
    data['exchange_rate'] = data.apply(lambda x: c.get_rate('EUR', 'USD', x.Date), axis=1)
    return data

def exchange_currency(data):
    data['Open_USD'] = data.apply(lambda x: x.Open*x.exchange_rate, axis=1)
    data['Close_USD'] = data.apply(lambda x: x.Close*x.exchange_rate, axis=1)
    data['CloseAdj_USD'] = data.apply(lambda x: x['Adj Close']*x.exchange_rate, axis=1)
    return data

def plot_price_history(data):
    sns.set_theme()
    sns.set_style('darkgrid')
    plt.figure(figsize=(16,8))
    plt.plot(data['Date'], data['Close_USD'], label='Close Price History', linewidth=4)
    plt.plot(data['Date'], data['Open_USD'], label='Open Price History')
    plt.plot(data['Date'] ,data['CloseAdj_USD'], label='Adj Close Price History')
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Price History in USD')
    plt.legend()
    plt.xticks(rotation=65)
    plt.title('Adidas Price History from March 2021 to March 2022')
    plt.savefig('./output/ADS_price_history.png')
    plt.show()

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


def main(filename):
    print('This should take roughly 5 minutes, please wait!')
    data = read_file(filename)
    data = currency_rates(data)
    data = exchange_currency(data)
    data.to_csv('./data/ads_preprocessing.csv')
    plot_price_history(data)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Please use the following command: python3 data_preprocessing.py ./data/ADS.DE.csv')
    else:
        main(sys.argv[1])