# Stock Price Prediction and Sentiment Analysis

This repository contains a web application for stock price prediction and sentiment analysis using Streamlit, LSTM models, and the Ollama. The application allows users to predict stock prices, compare stocks within sectors and industries, and analyze sentiments from stock-related news.

---

## Features

### 1. **Stock Price Prediction**
- Predict future stock prices using an LSTM-based deep learning model.
- Visualization of historical prices, daily returns, moving averages, and predictions.
- Adjustable prediction window (5–90 days).

### 2. **Sentiment Analysis**
- Fetch the latest news for selected stocks.
- Analyze sentiment of news headlines using Ollama's language model.
- Display sentiment alongside news articles.

### 3. **Stock Comparison**
- Compare average closing prices of stocks within a sector and industry.
- Highlight top-performing stocks.
- Display historical price trends and sentiment analysis for comparison.

### 4. **Educational Resources**
- Learn about key stock market metrics like P/E Ratio, Market Cap, Dividend Yield, and Technical Analysis.

---

## Installation

### Prerequisites
- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [YFinance](https://github.com/ranaroussi/yfinance)
- [Plotly](https://plotly.com/)
- [Ollama]

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-prediction-and-analysis.git
   cd stock-prediction-and-analysis
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run stock.py
   ```

---

## File Structure

### 1. **stock.py**
- Main script for the Streamlit application.
- Handles user interface, data visualization, and API interactions.

### 2. **new_model.py**
- Implements the LSTM model for stock price prediction.
- Contains utility functions for data preprocessing, model training, and future price predictions.

### 3. **/Stocks/**
- Contains sector and industry data in CSV format.
- Provides stock-specific historical data files.

---

## Usage

### **Stock Price Prediction**
1. Navigate to the "Predict Stock Price" menu.
2. Enter the stock ticker (e.g., `AAPL`) and select a prediction period.
3. View predicted stock prices and visualization.

### **Compare Stocks**
1. Navigate to the "Compare Stocks" menu.
2. Select a sector and industry.
3. Compare the top-performing stocks, view historical price trends, and analyze sentiment.

### **Educational Resources**
1. Navigate to the "Educational Resources" menu.
2. Learn about stock market metrics and analysis techniques.

---

## Technologies Used

- **Frontend**: Streamlit
- **Data Fetching**: YFinance
- **Modeling**: TensorFlow, Keras
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Sentiment Analysis**: Ollama 

---


