import streamlit as st
import pandas as pd
import os
import yfinance as yf
import plotly.graph_objects as go
import ollama
from new_model import main
# Function to load sector and industry data
def load_sector_industry_data():
    return pd.read_csv('./Stocks/sector_industries_vertical.csv')

# Function to load sector-specific stock data
def load_stock_data(sector):
    file_path = os.path.join('./Stocks', f'{sector}.csv')
    return pd.read_csv(file_path)

# Function to get company name from stock ticker
def get_company_name(ticker):
    stock_info = yf.Ticker(ticker).info
    return stock_info.get('shortName', 'Unknown Company')

# Fetch latest news related to a stock
def get_stock_news(ticker):
    stock_info = yf.Ticker(ticker)
    return stock_info.get_news()[:2]

# Sentiment analysis using OpenAI (Ollama)
def analyze_sentiment(article):
    response=ollama.generate(
        model='llama3.2:latest',
        prompt = f"Analyze the sentiment of this article: {article}"
    )
    print(response)
    return response['response']

# Display stock news with sentiment analysis
def display_stock_news(ticker):
    st.subheader(f"Latest News for {ticker}")
    news_list = get_stock_news(ticker)
    for article in news_list:
        st.write(f"**{article['title']}** - {article['publisher']}")
        sentiment = analyze_sentiment(article['title'])
        st.write(f"Sentiment: {sentiment}")
        st.write(f"[Read more]({article['link']})")
        st.write("---")

# Plot historical stock prices
def plot_stock_history(ticker):
    data = yf.download(ticker, period="1y")
    if data.empty:
        st.write(f"No data available for {ticker}.")
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data[('Close',f'{ticker}')], mode='lines+markers', name=f'{ticker} Close Price'))
    fig.update_layout(title=f'{ticker} Price Analysis', xaxis_title='Date', yaxis_title='Price')
    return fig

# Compare stocks with additional metrics
def compare_stocks(sector, industry):
    sector_data = load_stock_data(sector)
    stock_tickers = sector_data[industry].dropna().tolist()
    stock_performance = {}
    for ticker in stock_tickers:
        data = yf.download(ticker, period="1y")
        if not data.empty:
            stock_performance[ticker] = data[('Close',f"{ticker}")].mean()

    top_5_stocks = sorted(stock_performance, key=stock_performance.get, reverse=True)[:2]
    return [(get_company_name(ticker), ticker, stock_performance[ticker]) for ticker in top_5_stocks]

# Streamlit interface
st.set_page_config(page_title="Stock Prediction & Comparison", layout="wide")

def educational_resources():
    st.header("📚 Educational Resources")
    st.markdown("""
    - **P/E Ratio**: Price-to-Earnings Ratio.
    - **Market Cap**: Total market value of a company's shares.
    - **Dividend Yield**: Annual dividend divided by stock price.
    - **Technical Analysis**: Using charts for trading decisions.
    """)

def interactive_prediction_ui():
    stock_ticker = st.text_input("Enter Stock Ticker", placeholder="e.g., AAPL")
    prediction_days = st.slider("Select Prediction Period (Days)", 5, 90, 10)

    if st.button("Predict"):
        result,fig=main(stock_ticker,prediction_days)
        st.write("Predicted Prices:")
        st.write(result)
        st.plotly_chart(fig)
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=np.arange(len(last_prices)), y=last_prices, mode='lines+markers', name='Last Prices'))
        # fig.add_trace(go.Scatter(x=np.arange(len(last_prices), len(last_prices) + len(result)), y=result, mode='lines+markers', name='Predicted'))
        # st.plotly_chart(fig)

def compare_stocks_ui():
    sector_data = load_sector_industry_data()
    sector = st.selectbox("Select Sector", ["SELECT"] + list(sector_data.columns))

    if sector != "SELECT":
        industry = st.selectbox("Select Industry", ["SELECT"] + sector_data[sector].dropna().tolist())
        if industry != "SELECT" and st.button("Compare"):
            top_stocks = compare_stocks(sector, industry)
            for name, ticker, avg_close in top_stocks:
                st.write(f"**{name} ({ticker})** - Avg Close: {avg_close:.2f}")
                display_stock_news(ticker)
                st.plotly_chart(plot_stock_history(ticker))

# Main menu
st.title("📊 Stock Prediction & Analysis App")
choice = st.sidebar.selectbox("Menu", ["Predict Stock Price", "Compare Stocks", "Educational Resources"])

if choice == "Predict Stock Price":
    interactive_prediction_ui()
elif choice == "Compare Stocks":
    compare_stocks_ui()
elif choice == "Educational Resources":
    educational_resources()

########################## ABOVE IS ORIGINAL CODE #####################################################################
# import streamlit as st
# import pandas as pd
# import os
# import yfinance as yf
# import plotly.graph_objs as go
# from statsmodels.tsa.arima.model import ARIMA
# from keras._tf_keras.keras.models import Sequential
# from keras._tf_keras.keras.layers import LSTM, Dense
# import numpy as np
# import ollama

# # Function to load sector and industry data
# def load_sector_industry_data():
#     return pd.read_csv('./Stocks/sector_industries_vertical.csv')

# # Function to load sector-specific stock data
# def load_stock_data(sector):
#     file_path = os.path.join('./Stocks', f'{sector}.csv')
#     return pd.read_csv(file_path)

# # Function to get company name from stock ticker
# def get_company_name(ticker):
#     stock_info = yf.Ticker(ticker).info
#     return stock_info.get('shortName', 'Unknown Company')

# # Fetch latest news related to a stock
# @st.cache_data
# def fetch_news_with_sentiment(ticker):
#     stock_info = yf.Ticker(ticker)
#     news_list = stock_info.get_news()
#     if not news_list:
#         return []
    
#     # Extract titles for sentiment analysis
#     titles = [article['title'] for article in news_list[:5]]
#     sentiments = analyze_sentiment_batch(titles)
    
#     # Combine news articles with sentiments
#     news_with_sentiment = []
#     for article, sentiment in zip(news_list[:5], sentiments):
#         news_with_sentiment.append({
#             "title": article['title'],
#             "publisher": article['publisher'],
#             "link": article['link'],
#             "sentiment": sentiment
#         })
#     return news_with_sentiment

# # Sentiment analysis using OpenAI (Ollama)
# @st.cache_data
# def analyze_sentiment_batch(articles):
#     prompts = [f"Analyze the sentiment of this article: {article}" for article in articles]
#     responses = []
#     for prompt in prompts:
#         response = ollama.generate(
#             model='llama3.2:latest',
#             prompt=prompt
#         )
#         responses.append(response.get('response', 'Unknown'))
#     return responses

# # Display stock news with sentiment analysis
# def display_stock_news(ticker):
#     st.subheader(f"Latest News for {ticker}")
#     with st.spinner("Fetching news and analyzing sentiment..."):
#         news_list = fetch_news_with_sentiment(ticker)
#         if not news_list:
#             st.write("No news available for this stock.")
#             return
        
#         for article in news_list:
#             st.write(f"**{article['title']}** - {article['publisher']}")
#             st.write(f"Sentiment: {article['sentiment']}")
#             st.write(f"[Read more]({article['link']})")
#             st.write("---")

# # Plot historical stock prices
# def plot_stock_history(ticker):
#     data = yf.download(ticker, period="1y")
#     if data.empty:
#         st.write(f"No data available for {ticker}.")
#         return None

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines+markers', name=f'{ticker} Close Price'))
#     fig.update_layout(title=f'{ticker} Price Analysis', xaxis_title='Date', yaxis_title='Price')
#     return fig

# # Predict stock price using ARIMA
# def arima_predict(stock_ticker, steps=10):
#     data = yf.download(stock_ticker, period="5y")
#     model = ARIMA(data['Close'], order=(5, 1, 0))
#     result = model.fit()
#     return result.forecast(steps=steps), data['Close'][-10:].values

# # Prepare data for LSTM
# def prepare_lstm_data(data, look_back=60):
#     X, y = [], []
#     for i in range(look_back, len(data)):
#         X.append(data[i-look_back:i])
#         y.append(data[i])
#     return np.array(X), np.array(y)

# # Predict stock price using LSTM
# def lstm_predict(stock_ticker, steps=10):
#     data = yf.download(stock_ticker, period="5y")
#     close_prices = data['Close'].values
#     X, y = prepare_lstm_data(close_prices)
#     X = X.reshape((X.shape[0], X.shape[1], 1))

#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
#     model.add(LSTM(50))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.fit(X, y, epochs=10, batch_size=1, verbose=0)

#     predictions = []
#     last_sequence = close_prices[-60:]
#     for _ in range(steps):
#         input_data = last_sequence[-60:].reshape((1, 60, 1))
#         pred = model.predict(input_data)
#         predictions.append(pred[0][0])
#         last_sequence = np.append(last_sequence, pred[0][0])
#     return predictions

# # Combine ARIMA and LSTM predictions
# def predict_stock_price(stock_ticker, steps=10):
#     arima_pred, last_10_prices = arima_predict(stock_ticker, steps)
#     lstm_pred = lstm_predict(stock_ticker, steps)
#     combined_pred = [(ar + lst) / 2 for ar, lst in zip(arima_pred, lstm_pred)]
#     return combined_pred, last_10_prices

# # Compare stocks with additional metrics
# def compare_stocks(sector, industry):
#     sector_data = load_stock_data(sector)
#     stock_tickers = sector_data[industry].dropna().tolist()
#     stock_performance = {}
#     for ticker in stock_tickers:
#         data = yf.download(ticker, period="1y")
#         if not data.empty:
#             stock_performance[ticker] = data['Close'].mean()

#     top_5_stocks = sorted(stock_performance, key=stock_performance.get, reverse=True)[:5]
#     return [(get_company_name(ticker), ticker, stock_performance[ticker]) for ticker in top_5_stocks]

# # Streamlit interface
# st.set_page_config(page_title="Stock Prediction & Comparison", layout="wide")

# def educational_resources():
#     st.header("📚 Educational Resources")
#     st.markdown("""  
#     - **P/E Ratio**: Price-to-Earnings Ratio.
#     - **Market Cap**: Total market value of a company's shares.
#     - **Dividend Yield**: Annual dividend divided by stock price.
#     - **Technical Analysis**: Using charts for trading decisions.
#     """)

# def interactive_prediction_ui():
#     stock_ticker = st.text_input("Enter Stock Ticker", placeholder="e.g., AAPL")
#     prediction_days = st.slider("Select Prediction Period (Days)", 5, 60, 10)

#     if st.button("Predict"):
#         result, last_prices = predict_stock_price(stock_ticker, prediction_days)
#         st.write("Predicted Prices:")
#         for day, price in enumerate(result):
#             st.write(f"Day {day + 1}: {price:.2f}")

#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=np.arange(len(last_prices)), y=last_prices, mode='lines+markers', name='Last Prices'))
#         fig.add_trace(go.Scatter(x=np.arange(len(last_prices), len(last_prices) + len(result)), y=result, mode='lines+markers', name='Predicted'))
#         st.plotly_chart(fig)

# def compare_stocks_ui():
#     sector_data = load_sector_industry_data()
#     sector = st.selectbox("Select Sector", ["SELECT"] + list(sector_data.columns))

#     if sector != "SELECT":
#         industry = st.selectbox("Select Industry", ["SELECT"] + sector_data[sector].dropna().tolist())
#         if industry != "SELECT" and st.button("Compare"):
#             top_stocks = compare_stocks(sector, industry)
#             for name, ticker, avg_close in top_stocks:
#                 st.write(f"**{name} ({ticker})** - Avg Close: {avg_close:.2f}")
#                 display_stock_news(ticker)
#                 st.plotly_chart(plot_stock_history(ticker))

# # Main menu
# st.title("📊 Stock Prediction & Analysis App")
# choice = st.sidebar.selectbox("Menu", ["Predict Stock Price", "Compare Stocks", "Educational Resources"])

# if choice == "Predict Stock Price":
#     interactive_prediction_ui()
# elif choice == "Compare Stocks":
#     compare_stocks_ui()
# elif choice == "Educational Resources":
#     educational_resources()


# import streamlit as st
# import pandas as pd
# import os
# import yfinance as yf
# import plotly.graph_objs as go
# from statsmodels.tsa.arima.model import ARIMA
# from keras._tf_keras.keras.models import Sequential
# from keras._tf_keras.keras.layers import LSTM, Dense
# import numpy as np
# import ollama
# import concurrent.futures

# # Function to load sector and industry data
# @st.cache_data
# def load_sector_industry_data():
#     return pd.read_csv('./Stocks/sector_industries_vertical.csv')

# # Function to load sector-specific stock data
# @st.cache_data
# def load_stock_data(sector):
#     file_path = os.path.join('./Stocks', f'{sector}.csv')
#     return pd.read_csv(file_path)

# # Function to get company name from stock ticker
# @st.cache_data
# def get_company_name(ticker):
#     stock_info = yf.Ticker(ticker).info
#     return stock_info.get('shortName', 'Unknown Company')

# # Fetch latest news related to a stock
# @st.cache_data
# def fetch_news(ticker):
#     stock_info = yf.Ticker(ticker)
#     news_list = stock_info.get_news()
#     return news_list if news_list else []

# # Sentiment analysis using OpenAI (Ollama) - Optimized to process in parallel
# def analyze_sentiment_batch(articles):
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         prompts = [f"Analyze the sentiment of this article: {article}" for article in articles]
#         futures = [executor.submit(ollama.generate, model='llama3.2:latest', prompt=prompt) for prompt in prompts]
#         results = [future.result() for future in concurrent.futures.as_completed(futures)]
#         return [result.get('response', 'Unknown') for result in results]

# # Fetch news and perform sentiment analysis efficiently
# @st.cache_data
# def fetch_news_with_sentiment(ticker):
#     news_list = fetch_news(ticker)
#     if not news_list:
#         return []
    
#     # Extract titles for sentiment analysis
#     titles = [article['title'] for article in news_list[:2]]
#     sentiments = analyze_sentiment_batch(titles)
    
#     # Combine news articles with sentiments
#     news_with_sentiment = [
#         {
#             "title": article['title'],
#             "publisher": article['publisher'],
#             "link": article['link'],
#             "sentiment": sentiment
#         }
#         for article, sentiment in zip(news_list[:1], sentiments)
#     ]
#     return news_with_sentiment

# # Display stock news with sentiment analysis
# def display_stock_news(ticker):
#     st.subheader(f"Latest News for {ticker}")
#     with st.spinner("Fetching news and analyzing sentiment..."):
#         news_list = fetch_news_with_sentiment(ticker)
#         if not news_list:
#             st.write("No news available for this stock.")
#             return
        
#         for article in news_list:
#             st.write(f"**{article['title']}** - {article['publisher']}")
#             st.write(f"Sentiment: {article['sentiment']}")
#             st.write(f"[Read more]({article['link']})")
#             st.write("---")

# # Plot historical stock prices
# @st.cache_data
# def plot_stock_history(ticker):
#     data = yf.download(ticker, period="1y")
#     if data.empty:
#         st.write(f"No data available for {ticker}.")
#         return None

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines+markers', name=f'{ticker} Close Price'))
#     fig.update_layout(title=f'{ticker} Price Analysis', xaxis_title='Date', yaxis_title='Price')
#     return fig

# # Predict stock price using ARIMA
# @st.cache_data
# def arima_predict(stock_ticker, steps=10):
#     data = yf.download(stock_ticker, period="5y")
#     model = ARIMA(data['Close'], order=(5, 1, 0))
#     result = model.fit()
#     return result.forecast(steps=steps), data['Close'][-10:].values

# # Prepare data for LSTM
# def prepare_lstm_data(data, look_back=60):
#     X, y = [], []
#     for i in range(look_back, len(data)):
#         X.append(data[i-look_back:i])
#         y.append(data[i])
#     return np.array(X), np.array(y)

# # Predict stock price using LSTM
# @st.cache_data
# def lstm_predict(stock_ticker, steps=10):
#     data = yf.download(stock_ticker, period="5y")
#     close_prices = data['Close'].values
#     X, y = prepare_lstm_data(close_prices)
#     X = X.reshape((X.shape[0], X.shape[1], 1))

#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
#     model.add(LSTM(50))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.fit(X, y, epochs=10, batch_size=1, verbose=0)

#     predictions = []
#     last_sequence = close_prices[-60:]
#     for _ in range(steps):
#         input_data = last_sequence[-60:].reshape((1, 60, 1))
#         pred = model.predict(input_data)
#         predictions.append(pred[0][0])
#         last_sequence = np.append(last_sequence, pred[0][0])
#     return predictions

# # Combine ARIMA and LSTM predictions
# def predict_stock_price(stock_ticker, steps=10):
#     arima_pred, last_10_prices = arima_predict(stock_ticker, steps)
#     lstm_pred = lstm_predict(stock_ticker, steps)
#     combined_pred = [(ar + lst) / 2 for ar, lst in zip(arima_pred, lstm_pred)]
#     return combined_pred, last_10_prices

# # Compare stocks with additional metrics
# @st.cache_data
# def compare_stocks(sector, industry):
#     sector_data = load_stock_data(sector)
#     stock_tickers = sector_data[industry].dropna().tolist()
#     stock_performance = {}
#     for ticker in stock_tickers:
#         data = yf.download(ticker, period="1y")
#         if not data.empty:
#             stock_performance[ticker] = data['Close'].mean()

#     top_5_stocks = sorted(stock_performance, key=stock_performance.get, reverse=True)[:2]
#     return [(get_company_name(ticker), ticker, stock_performance[ticker]) for ticker in top_5_stocks]

# # Streamlit interface
# st.set_page_config(page_title="Stock Prediction & Comparison", layout="wide")

# def educational_resources():
#     st.header("📚 Educational Resources")
#     st.markdown("""  
#     - **P/E Ratio**: Price-to-Earnings Ratio.
#     - **Market Cap**: Total market value of a company's shares.
#     - **Dividend Yield**: Annual dividend divided by stock price.
#     - **Technical Analysis**: Using charts for trading decisions.
#     """)

# def interactive_prediction_ui():
#     stock_ticker = st.text_input("Enter Stock Ticker", placeholder="e.g., AAPL")
#     prediction_days = st.slider("Select Prediction Period (Days)", 5, 60, 10)

#     if st.button("Predict"):
#         result, last_prices = predict_stock_price(stock_ticker, prediction_days)
#         st.write("Predicted Prices:")
#         for day, price in enumerate(result):
#             st.write(f"Day {day + 1}: {price:.2f}")

#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=np.arange(len(last_prices)), y=last_prices, mode='lines+markers', name='Last Prices'))
#         fig.add_trace(go.Scatter(x=np.arange(len(last_prices), len(last_prices) + len(result)), y=result, mode='lines+markers', name='Predicted'))
#         st.plotly_chart(fig)

# def compare_stocks_ui():
#     sector_data = load_sector_industry_data()
#     sector = st.selectbox("Select Sector", ["SELECT"] + list(sector_data.columns))

#     if sector != "SELECT":
#         industry = st.selectbox("Select Industry", ["SELECT"] + sector_data[sector].dropna().tolist())
#         if industry != "SELECT" and st.button("Compare"):
#             top_stocks = compare_stocks(sector, industry)
#             for name, ticker, avg_close in top_stocks:
#                 st.write(f"**{name} ({ticker})** - Avg Close: {avg_close:.2f}")
#                 display_stock_news(ticker)
#                 st.plotly_chart(plot_stock_history(ticker))

# # Main menu
# st.title("📊 Stock Prediction & Analysis App")
# choice = st.sidebar.selectbox("Menu", ["Predict Stock Price", "Compare Stocks", "Educational Resources"])

# if choice == "Predict Stock Price":
#     interactive_prediction_ui()
# elif choice == "Compare Stocks":
#     compare_stocks_ui()
# elif choice == "Educational Resources":
#     educational_resources()












# import streamlit as st
# import pandas as pd
# import os
# import yfinance as yf
# import plotly.graph_objs as go
# from statsmodels.tsa.arima.model import ARIMA
# from keras._tf_keras.keras.models import Sequential
# from keras._tf_keras.keras.layers import LSTM, Dense
# import numpy as np
# import ollama
# import concurrent.futures

# # Function to load sector and industry data
# @st.cache_data
# def load_sector_industry_data():
#     return pd.read_csv('./Stocks/sector_industries_vertical.csv')

# # Function to load sector-specific stock data
# @st.cache_data
# def load_stock_data(sector):
#     file_path = os.path.join('./Stocks', f'{sector}.csv')
#     return pd.read_csv(file_path)

# # Function to get company name from stock ticker
# @st.cache_data
# def get_company_name(ticker):
#     stock_info = yf.Ticker(ticker).info
#     return stock_info.get('shortName', 'Unknown Company')

# # Fetch stock data for previous performance summary
# @st.cache_data
# def fetch_stock_data(ticker):
#     return yf.download(ticker, period="1y")

# # Generate a description of stock's performance using Ollama
# def generate_performance_description(ticker):
#     stock_data = fetch_stock_data(ticker)
#     if stock_data.empty:
#         return "No data available for this stock."

#     # Get the percentage change in stock's close price over the past year
#     start_price = stock_data['Close'].iloc[0]
#     end_price = stock_data['Close'].iloc[-1]
#     change = ((end_price - start_price) / start_price) * 100

#     # Create a summary prompt for Ollama
#     performance_summary = f"Analyze the performance of {ticker} over the last year. " \
#                           f"The stock started at {start_price:.2f} and ended at {end_price:.2f}. " \
#                           f"It changed by {change:.2f}%."\
#                           f"Generate a two line short summary by analyzing all the above points."

#     # Get the sentiment analysis from Ollama
#     response = ollama.generate(
#         model="llama3.2:latest",
#         prompt=performance_summary
#     )

#     return response.get("response", "Unable to generate performance summary.")

# # Plot historical stock prices
# @st.cache_data
# def plot_stock_history(ticker):
#     data = yf.download(ticker, period="1y")
#     if data.empty:
#         st.write(f"No data available for {ticker}.")
#         return None

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines+markers', name=f'{ticker} Close Price'))
#     fig.update_layout(title=f'{ticker} Price Analysis', xaxis_title='Date', yaxis_title='Price')
#     return fig

# # Predict stock price using ARIMA
# @st.cache_data
# def arima_predict(stock_ticker, steps=10):
#     data = yf.download(stock_ticker, period="5y")
#     model = ARIMA(data['Close'], order=(5, 1, 0))
#     result = model.fit()
#     return result.forecast(steps=steps), data['Close'][-10:].values

# # Prepare data for LSTM
# def prepare_lstm_data(data, look_back=60):
#     X, y = [], []
#     for i in range(look_back, len(data)):
#         X.append(data[i-look_back:i])
#         y.append(data[i])
#     return np.array(X), np.array(y)

# # Predict stock price using LSTM
# @st.cache_data
# def lstm_predict(stock_ticker, steps=10):
#     data = yf.download(stock_ticker, period="5y")
#     close_prices = data['Close'].values
#     X, y = prepare_lstm_data(close_prices)
#     X = X.reshape((X.shape[0], X.shape[1], 1))

#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
#     model.add(LSTM(50))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.fit(X, y, epochs=10, batch_size=1, verbose=0)

#     predictions = []
#     last_sequence = close_prices[-60:]
#     for _ in range(steps):
#         input_data = last_sequence[-60:].reshape((1, 60, 1))
#         pred = model.predict(input_data)
#         predictions.append(pred[0][0])
#         last_sequence = np.append(last_sequence, pred[0][0])
#     return predictions

# # Combine ARIMA and LSTM predictions
# def predict_stock_price(stock_ticker, steps=10):
#     arima_pred, last_10_prices = arima_predict(stock_ticker, steps)
#     lstm_pred = lstm_predict(stock_ticker, steps)
#     combined_pred = [(ar + lst) / 2 for ar, lst in zip(arima_pred, lstm_pred)]
#     return combined_pred, last_10_prices

# # Compare stocks with additional metrics
# @st.cache_data
# def compare_stocks(sector, industry):
#     sector_data = load_stock_data(sector)
#     stock_tickers = sector_data[industry].dropna().tolist()
#     stock_performance = {}
#     for ticker in stock_tickers:
#         data = yf.download(ticker, period="1y")
#         if not data.empty:
#             stock_performance[ticker] = data['Close'].mean()

#     top_5_stocks = sorted(stock_performance, key=stock_performance.get, reverse=True)[:2]
#     return [(get_company_name(ticker), ticker, stock_performance[ticker]) for ticker in top_5_stocks]

# # Streamlit interface
# st.set_page_config(page_title="Stock Prediction & Comparison", layout="wide")

# def educational_resources():
#     st.header("📚 Educational Resources")
#     st.markdown("""  
#     - **P/E Ratio**: Price-to-Earnings Ratio.
#     - **Market Cap**: Total market value of a company's shares.
#     - **Dividend Yield**: Annual dividend divided by stock price.
#     - **Technical Analysis**: Using charts for trading decisions.
#     """)

# def interactive_prediction_ui():
#     stock_ticker = st.text_input("Enter Stock Ticker", placeholder="e.g., AAPL")
#     prediction_days = st.slider("Select Prediction Period (Days)", 5, 60, 10)

#     if st.button("Predict"):
#         result, last_prices = predict_stock_price(stock_ticker, prediction_days)
#         st.write("Predicted Prices:")
#         for day, price in enumerate(result):
#             st.write(f"Day {day + 1}: {price:.2f}")

#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=np.arange(len(last_prices)), y=last_prices, mode='lines+markers', name='Last Prices'))
#         fig.add_trace(go.Scatter(x=np.arange(len(last_prices), len(last_prices) + len(result)), y=result, mode='lines+markers', name='Predicted'))
#         st.plotly_chart(fig)

# def compare_stocks_ui():
#     sector_data = load_sector_industry_data()
#     sector = st.selectbox("Select Sector", ["SELECT"] + list(sector_data.columns))

#     if sector != "SELECT":
#         industry = st.selectbox("Select Industry", ["SELECT"] + sector_data[sector].dropna().tolist())
#         if industry != "SELECT" and st.button("Compare"):
#             top_stocks = compare_stocks(sector, industry)
#             for name, ticker, avg_close in top_stocks:
#                 st.write(f"**{name} ({ticker})** - Avg Close: {avg_close:.2f}")
#                 performance_description = generate_performance_description(ticker)
#                 st.write(f"**Performance Summary**: {performance_description}")
#                 st.plotly_chart(plot_stock_history(ticker))

# # Main menu
# st.title("📊 Stock Prediction & Analysis App")
# choice = st.sidebar.selectbox("Menu", ["Predict Stock Price", "Compare Stocks", "Educational Resources"])

# if choice == "Predict Stock Price":
#     interactive_prediction_ui()
# elif choice == "Compare Stocks":
#     compare_stocks_ui()
# elif choice == "Educational Resources":
#     educational_resources()


