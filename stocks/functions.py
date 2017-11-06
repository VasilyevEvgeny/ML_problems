
# coding: utf-8

# In[7]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

def make_stocks_dict(df):
    stocks = {}
    for i in range(len(df)):
        number = df.loc[i, 'number']
        if number in stocks.keys():
            prev = stocks[number]
            current = (df.loc[i, 'day'], df.loc[i, 'percent'])
            if type(prev) == tuple:
                res = [prev, current]
            else:
                prev.append(current)
                res = prev[:]
            stocks[number] = res  
        else:
            stocks.update({number:(df.loc[i, 'day'], df.loc[i, 'percent'])})
            
    return stocks

def plot_frequencies(stocks):
    plt.figure(figsize=(15,5))
    lengths = [ len(val) for val in stocks.values() ]
    X = np.unique(lengths)
    Y = [ lengths.count(i) for i in X ]
    plt.stem(X, Y, 'r')
    plt.plot(X, Y, marker='o', markersize=10, linestyle='None', color='red')
    plt.style.use('classic')
    plt.xticks(np.arange(min(lengths)-2, max(lengths)+2))
    plt.yticks(np.arange(min(Y)-2, max(Y)+2))
    plt.xlabel('stock length')
    plt.ylabel('number of stocks')
    plt.show()
    
def plot_stocks(stocks):
    for number in stocks.keys():
        plt.figure(figsize=(15,3))
        X, Y = [], []
        for pair in stocks[number]:
            X.append(pair[0])
            Y.append(pair[1])
        print('checking sum of percents:', sum(Y))
        plt.plot(X, Y, color='black', marker='o', markersize=10, linewidth=5)
        plt.style.use('bmh')
        title = 'stock number = ' + str(number)
        plt.title(title, loc='center')
        plt.xticks(np.arange(len(stocks[number])+2))
        plt.xlabel('day')
        plt.ylabel('percent')
        plt.show()
        
def calibrate_percents(stocks):
    for key in stocks.keys():
        percents_sum = 0
        for pair in stocks[key]:
            percents_sum += pair[1]
        delta = (percents_sum - 1.0) / len(stocks[key])
        new_val = []
        for pair in stocks[key]:
            new_pair1 = pair[1] - delta
            new_val.append((pair[0], new_pair1))
        stocks[key] = new_val  
    return stocks

def transform_stocks(stocks):
    stocks_transformed = {}

    lengths = [ len(val) for val in stocks.values() ]
    keys = np.unique(lengths)
    vals = [ lengths.count(i) for i in keys ]
    freqs = dict(zip(keys, vals))

    keys = np.unique([ len(val) for val in stocks.values() ])
    for key in stocks.keys():
        number_of_days = len(stocks[key])
        if number_of_days not in stocks_transformed.keys():
            new_val = []
            for pair in stocks[key]:
                new_val.append((pair[0], pair[1] / freqs[number_of_days]))
            stocks_transformed.update({number_of_days:new_val})
        else:
            old_percents = []
            for i in range(len(stocks[key])):
                old_percents.append(stocks[key][i][1])
            new_val = []
            for i in range(len(stocks_transformed[number_of_days])):
                new_val.append((stocks_transformed[number_of_days][i][0],                                 old_percents[i] + stocks_transformed[number_of_days][i][1] / freqs[number_of_days]))
            stocks_transformed.update({number_of_days:new_val})

    for key in stocks_transformed.keys():
        new_val = []
        for pair in stocks_transformed[key]:
            new_val.append((pair[0] / len(stocks_transformed[key]), pair[1]))
        stocks_transformed[key] = new_val
    
    return calibrate_percents(stocks_transformed)

def plot_stocks_transformed(stocks_transformed):
    for key in stocks_transformed.keys():
        plt.figure(figsize=(15,3))
        X, Y = [], []
        for pair in stocks_transformed[key]:
            X.append(pair[0])
            Y.append(pair[1])
        print('checking sum of percents:', sum(Y))
        plt.plot(X, Y, color='black', marker='o', markersize=10, linewidth=5)
        plt.style.use('bmh')
        title = 'stock length = ' + str(key)
        plt.title(title, loc='center')
        plt.xlim(0,1.05)
        plt.xlabel('day')
        plt.ylabel('percent')
        plt.show()
        
def make_weights_list(stock_length, lengths, func='exp'):
    weights_list = []
    if func=='exp':
        delta = 3. * (max(lengths) - min(lengths))
        for i in lengths:
            weights_list.append(np.exp(-np.abs(stock_length-i) / delta))
    delta = (sum(weights_list) - 1.0) / len(weights_list)
    
    return [ w - delta if delta > 0 else w + delta for w in weights_list ]

def linear_approximation(x1, y1, x2, y2, x):
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    return k * x + b

def predict_stock(stock_length):
    df = pd.read_csv('stocks.csv', sep=',', names=['number', 'day', 'percent'], header=0)
    stocks = make_stocks_dict(df)
    stocks_transformed = transform_stocks(stocks)
    
    prediction = []
    weights = make_weights_list(stock_length, list(stocks_transformed.keys()))
    x_coords = [ (i+1) / stock_length for i in range(stock_length) ]
    for n, x_coord in enumerate(x_coords):
        percents = 0.
        for idx, key in enumerate(stocks_transformed.keys()):
            for i in range(1,len(stocks_transformed[key])):
                flag_done = 0
                if stocks_transformed[key][i][0] >= x_coord:
                    x1, x2 = stocks_transformed[key][i][0], stocks_transformed[key][i-1][0]
                    y1, y2 = stocks_transformed[key][i][1], stocks_transformed[key][i-1][1]
                    flag_done = 1
                    break
                if not flag_done:
                    end = len(stocks_transformed[key])-1
                    x1, x2 = stocks_transformed[key][end][0], stocks_transformed[key][end-1][0]
                    y1, y2 = stocks_transformed[key][end][1], stocks_transformed[key][end-1][1] 
            y_coord = linear_approximation(x1, y1, x2, y2, x_coord)
            percents += weights[idx] * y_coord
        prediction.append((n+1, percents))
    pred_dict = {stock_length: prediction}
    pred_dict = calibrate_percents(pred_dict)
    df_prediction = pd.DataFrame(columns=['day', 'percent'])
    for idx, pair in enumerate(pred_dict[stock_length]):
        df_prediction.loc[idx, 'day'] = pair[0]
        df_prediction.loc[idx, 'percent'] = pair[1]
    
    return df_prediction

def plot_predictions():
    stock_lengths = list(range(1,51))
    for stock_length in stock_lengths:
        df_prediction = predict_stock(stock_length)
        plt.figure(figsize=(15,3))
        X = list(df_prediction['day'])
        Y = list(df_prediction['percent'])
        print('checking sum of percents:', sum(Y))
        plt.plot(X, Y, color='black', marker='o', markersize=10, linewidth=5)
        plt.style.use('bmh')
        title = 'stock length = ' + str(len(df_prediction))
        plt.title(title, loc='center')
        plt.xlim(min(X)-1, max(X)+1)
        plt.xticks(np.arange(stock_length+2))
        plt.xlabel('day')
        plt.ylabel('percent')
        plt.show()


# In[ ]:



