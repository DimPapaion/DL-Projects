
import matplotlib.pyplot as plt
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly.graph_objects as go
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Utils import *
from Train_Test import *
from Models import *

def main():
    ETH = pd.read_csv('ETH-USD.csv')  # Read data from csv.

    print(ETH.tail(), ETH.info() )


    # A simple visualization of data.
    ETH.plot(x="Date", y="Close", figsize=(18, 6), legend=True, use_index=True)
    plt.title('Ethereum Historical Price')
    plt.xlabel('Date')
    plt.ylabel('Ether to USD')
    plt.legend()
    plt.show()

    ETH['Date'] = pd.to_datetime(ETH['Date'])  # Convert to daytime index.
    ETH.set_index(ETH['Date'], inplace=True)
    ETH.info()

    # Make Candlestick visualization
    configure_plotly_browser_state()
    init_notebook_mode(connected=False)
    figure = go.Figure(
        data=[go.Candlestick(
            x=ETH.Date,
            low=ETH['Low'],
            high=ETH['High'],
            close=ETH['Close'],
            open=ETH['Open'],
            increasing_line_color='green',
            decreasing_line_color='red')])

    figure.update_layout(xaxis_rangeslider_visible=False,
                         title='Ether Prices',
                         yaxis_title='Ether Price in USD ($)',
                         xaxis_title='Date')
    figure.show()

    # Normalize Data and Spliting into Train Valid Test set.

    ETH['Close'].isnull().values.any()

    ETH['Close'].isnull().values.sum()

    ETH = ETH[ETH['Close'].notna()]

    price = ETH[['Close']]
    print(price)

    # Normalize the Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    price = scaler.fit_transform(ETH['Close'].values.reshape(-1, 1))
    print(price)

    # Persentage of valid and test split
    valid_num = 20
    test_num = 20
    seq_len = 101  # looking back N steps

    x_train, y_train, x_valid, y_valid, x_test, y_test = make_seq(price, seq_len)
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_valid.shape = ', x_valid.shape)
    print('y_valid.shape = ', y_valid.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)

    # Preparing Data for LSTM

    # Transfrom from numpy to tensors
    X_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)

    X_val = torch.Tensor(x_valid)
    y_val = torch.Tensor(y_valid)

    X_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    print(X_val.shape)
    print(y_val.shape)

    # Make tensorDataset
    train = TensorDataset(X_train, y_train)
    valid = TensorDataset(X_val, y_val)
    test = TensorDataset(X_test, y_test)

    # Create the DataLoaders.
    batch_size = 64
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    valid_dl = DataLoader(valid, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dl_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)


    # Initialize LSTM Model
    input_dim = 1
    output_dim = 1
    hidden_dim = 185
    layer_dim = 2
    n_epochs = 150
    learning_rate = 0.001

    model_lstm = LSTMModel(input_dim=input_dim,
                           hidden_dim=hidden_dim,
                           layer_dim=layer_dim,
                           output_dim=output_dim)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=learning_rate)

    print(model_lstm)

    # Train it
    train_loss, valid_loss = fit_train(n_epochs=n_epochs, train_dl=train_dl, valid_dl=valid_dl,
                                       model=model_lstm, loss_function=loss_function, optimizer=optimizer)

    # Figure for loss
    plt.figure(figsize=(15, 6))
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.xlabel("epochs")
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. Number of Epochs LSTM Model');

    predictions, values = test_fit(model=model_lstm, test_dl_one=test_dl_one)  # Test the model

    df_result = format_predictions(predictions, values, ETH[1558:], scaler)  # Show Dataset
    df_result

    # Make plotly for the results.
    configure_plotly_browser_state()
    # Set notebook mode to work in offline
    pyo.init_notebook_mode()

    plot_predictions(df_result, model_name="lstm")

