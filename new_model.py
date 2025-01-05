import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
import yfinance as yf
from datetime import datetime
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
def main(ticker, prediction_days):
    # Download stock data using yfinance
    tech_list = [ticker]
    company_list = []
    company_name = []

    for stock in tech_list:
        stock_data = yf.download(stock, period="10y")
        company_list.append(stock_data)
        company_name.append(stock)

    # Add a company name column to each DataFrame
    for company, com_name in zip(company_list, company_name):
        company['company_name'] = com_name

    # Concatenate all DataFrames into one
    df = pd.concat(company_list, axis=0)
    df.columns = df.columns.droplevel('Ticker')
    print(f"COLUMNS:{df.columns}")
    # Display the last 10 rows of the combined data
    print(df.tail(10))
    print(df.describe())

    # Plotting adjusted close price
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company['Adj Close'].plot()
        plt.ylabel('Adj Close')
        plt.xlabel(None)
        plt.title(f"Closing Price of {tech_list[i - 1]}")

    plt.tight_layout()
    plt.show()

    # Plotting volume data
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company['Volume'].plot()
        plt.ylabel('Volume')
        plt.xlabel(None)
        plt.title(f"Sales Volume for {tech_list[i - 1]}")

    plt.tight_layout()
    plt.show()

    # Adding moving averages
    ma_day = [10, 20, 50]

    for ma in ma_day:
        for company in company_list:
            column_name = f"MA for {ma} days"
            company[column_name] = company['Adj Close'].rolling(ma).mean()

    # Plotting moving averages
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    # Access the first company data
    company = company_list[0]

    company[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0, 0])
    axes[0, 0].set_title(f'{tech_list[0]} Moving Averages')

    fig.tight_layout()
    plt.show()

    for company in company_list:
        company['Daily Return'] = company['Adj Close'].pct_change()

    # Plot daily returns
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    company['Daily Return'].plot(ax=axes[0, 0], legend=True, linestyle='--', marker='o')
    axes[0, 0].set_title('Daily Return for ' + tech_list[0])
    fig.tight_layout()
    plt.show()

    # Display histogram of daily returns
    plt.figure(figsize=(12, 9))
    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company['Daily Return'].hist(bins=50)
        plt.xlabel('Daily Return')
        plt.ylabel('Counts')
        plt.title(f'{company_name[i - 1]}')
    plt.tight_layout()
    plt.show()

    # Preparing data for LSTM model
    print(f"COLUMNS:{df.columns}")
    data = df.filter(['Close'])
    dataset = data.values
    print(f"DATASET:{dataset}")
    training_data_len = int(np.ceil(len(dataset) * 0.95))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Testing the model
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the RMSE value
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print(f'RMSE: {rmse}')

    # Visualize the results
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    plt.figure(figsize=(16, 6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price INR (₹)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

    # Future Predictions
    last_60_days = scaled_data[-60:]
    future_predictions = []

    for _ in range(prediction_days):
        x_future = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
        future_price = model.predict(x_future)
        future_predictions.append(future_price[0, 0])
        last_60_days = np.append(last_60_days[1:], future_price, axis=0)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Predict future dates
    last_date = df.index[-1]
    date_range = pd.date_range(last_date, periods=prediction_days + 1, freq='B')[1:]

    future_df = pd.DataFrame(future_predictions, index=date_range, columns=["Predictions"])
    trace_actual = go.Scatter(x=stock_data.index, y=stock_data[('Close',f'{ticker}')], mode='lines', name='Actual Price')
    trace_future = go.Scatter(x=future_df.index, y=future_df['Predictions'], mode='lines', name='Future Predictions', line=dict(dash='dash'))

    # Display the future predictions
    print(future_df)
    layout = go.Layout(
            title=f'{ticker} Stock Price vs Predictions',
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price"),
            template='plotly_dark',
        )

    fig = go.Figure(data=[trace_actual, trace_future], layout=layout)

    return future_df,fig

if __name__ == "__main__":
    # ticker = input("Enter the stock ticker (e.g., SBIN.BO): ")
    # prediction_days = int(input("Enter the number of days to predict: "))
    main()

