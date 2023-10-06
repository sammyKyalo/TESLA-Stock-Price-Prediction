import streamlit as st
import pandas as pd 
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from ta.trend import SMAIndicator
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima as pm

# Set page title and favicon
st.set_page_config(page_title='TESLA Stock Analysis', page_icon=':chart_with_upwards_trend:')

# Title of the app
st.title('TESLA Stock Analysis and Prediction')

# Sidebar for user input
st.sidebar.header('User Input')
ticker = st.sidebar.text_input('Enter Ticker Symbol', 'TSLA')
start_date_input = st.sidebar.date_input('Start Date', datetime.today() - timedelta(days=4*365))
end_date_input = st.sidebar.date_input('End Date', datetime.today())

# Fetch stock data
try:
    data = yf.download(ticker, start=start_date_input, end=end_date_input)
    st.sidebar.success(f'Successfully loaded data for {ticker}')
except Exception as e:
    st.sidebar.error(f'Error loading data: {e}')
    st.stop()

# Candlestick chart 
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    subplot_titles=['Candlestick Chart', 'Moving Average'])
fig.add_trace(go.Candlestick(x=data.index, 
                             open=data['Open'], 
                             high=data['High'], 
                             low=data['Low'], 
                             close=data['Close'], 
                             name='Candlesticks'), row=1, col=1)

# Moving Average
period = st.sidebar.slider('Select Moving Average Period', min_value=5, max_value=50, step=5, value=20)
data['SMA'] = SMAIndicator(data['Close'], window=period).sma_indicator()
fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], mode='lines', name=f'{period} days SMA', line=dict(color='orange')), row=2, col=1)

# Set layout for Candlestick Chart
fig.update_xaxes(type='category')
fig.update_layout(xaxis_rangeslider_visible=False, height=600, title_text=f'{ticker} Stock Analysis')
st.plotly_chart(fig)

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Additional Analysis
st.header('Additional Analysis')

# Calculate and display moving average
ma_period = st.slider('Select Moving Average Period for Analysis', min_value=5, max_value=50, step=5, value=20)
data['MA'] = data['Close'].rolling(window=ma_period).mean()
st.write(f'{ma_period} days Moving Average:')
st.line_chart(data['MA'])

# Calculate and display Bollinger Bands
bollinger_window = st.slider('Select Bollinger Bands Window', min_value=5, max_value=50, step=5, value=20)
data['rolling_std'] = data['Close'].rolling(window=bollinger_window).std()
data['upper_band'] = data['MA'] + 2 * data['rolling_std']
data['lower_band'] = data['MA'] - 2 * data['rolling_std']
st.write('Bollinger Bands:')
st.line_chart(data[['upper_band', 'MA', 'lower_band']])

# Calculate and display Relative Strength Index (RSI)
rsi_window = st.slider('Select RSI Window', min_value=5, max_value=50, step=5, value=14)
delta = data['Close'].diff(1)
delta = delta.dropna()
up = delta.copy()
down = delta.copy()
up[up < 0] = 0
down[down > 0] = 0
avg_gain = up.rolling(window=rsi_window).mean()
avg_loss = abs(down.rolling(window=rsi_window).mean())
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
st.write('Relative Strength Index (RSI):')
st.line_chart(rsi)

# Add trading signals based on RSI
overbought_level = st.slider('Select Overbought Level', min_value=50, max_value=70, step=1, value=70)
oversold_level = st.slider('Select Oversold Level', min_value=30, max_value=50, step=1, value=30)
data['RSI'] = rsi
data['Buy_Signal'] = data['RSI'] < oversold_level
data['Sell_Signal'] = data['RSI'] > overbought_level

# Plot Buy and Sell signals
fig_signals = go.Figure(data=[go.Candlestick(x=data.index, 
                                             open=data['Open'], 
                                             high=data['High'], 
                                             low=data['Low'], 
                                             close=data['Close'], 
                                             name='Candlesticks')])
fig_signals.add_trace(go.Scatter(x=data.index, y=data['MA'], mode='lines', name=f'{ma_period} days SMA', line=dict(color='orange')))
fig_signals.add_trace(go.Scatter(x=data[data['Buy_Signal']].index, 
                                 y=data[data['Buy_Signal']]['Low'], 
                                 mode='markers', 
                                 marker=dict(color='green', size=10), 
                                 name='Buy Signal'))
fig_signals.add_trace(go.Scatter(x=data[data['Sell_Signal']].index, 
                                 y=data[data['Sell_Signal']]['High'], 
                                 mode='markers', 
                                 marker=dict(color='red', size=10), 
                                 name='Sell Signal'))
fig_signals.update_xaxes(type='category')
fig_signals.update_layout(xaxis_rangeslider_visible=False, height=600, title_text=f'{ticker} Stock Analysis with Signals')
st.plotly_chart(fig_signals)


try:
    data = yf.download(ticker, start=start_date_input, end=end_date_input)
    data = data.asfreq("B")
    st.sidebar.success(f'Successfully loaded data for {ticker}')
except Exception as e:
    st.sidebar.error(f'Error loading data: {e}')
    st.stop()

import streamlit as st
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm

# Set the title of the Streamlit app
st.title('TSLA Stock Price Forecast')

tickers = ["TSLA"]

end_date = datetime.today()
start_date = end_date - timedelta(days = 4*365)

close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start=start_date,end=end_date)
    close_df[ticker] = data['Close']

df= close_df

df = df.asfreq("b")
df = df.fillna(method="ffill")

df['Returns'] = df.TSLA.pct_change()
df['lag_1'] = df.TSLA.shift(1)



p, d, q = (2, 1, 2)  
model = ARIMA(df.TSLA.dropna(), order=(p, d, q))
results = model.fit()

n = 5  
forecast = results.forecast(steps=n)

start_date = "2023-09-25"
end_date = "2023-10-03"
df_pred = results.predict(start=start_date, end=end_date)

actual_prices = df['TSLA']['2019-09-24':'2023-09-22']

# Display the forecasted prices in a table
st.subheader('Forecasted Prices')
st.write(forecast.to_frame())

# Display the plot for actual prices and predictions
st.subheader('Actual TSLA Prices vs. Predictions')
fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(actual_prices, label='Actual TSLA Prices', color='blue')
ax.plot(df_pred, label='Predictions', color='red')
ax.set_title("Actual TSLA Prices vs. Predictions")
ax.set_xlabel("Date",fontsize=14)
ax.set_ylabel("Price",fontsize=14)
plt.tight_layout()
plt.legend()
st.pyplot(fig)

