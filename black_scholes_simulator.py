import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from numpy import log, sqrt, exp

# Streamlit page settings
st.set_page_config(page_title="📈 Option Pricing Explorer", layout="wide")

# Black-Scholes Pricing Class
class BlackScholesModel:
    def __init__(self, S, K, T, sigma, r):
        self.S = S
        self.K = K
        self.T = T
        self.sigma = sigma
        self.r = r

    def price(self):
        d1 = (log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * sqrt(self.T))
        d2 = d1 - self.sigma * sqrt(self.T)

        call = self.S * norm.cdf(d1) - self.K * exp(-self.r * self.T) * norm.cdf(d2)
        put = self.K * exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

        return call, put

# GUI - User Inputs
st.sidebar.title("Option Input Settings")

spot_price = st.sidebar.number_input("Current Price (S)", value=50.0)
strike_price = st.sidebar.number_input("Strike Price (K)", value=50.0)
time_to_expiry = st.sidebar.number_input("Time to Expiry (Years)", value=1.0)
volatility = st.sidebar.number_input("Volatility (σ)", value=0.2)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05)

st.sidebar.markdown("---")
st.sidebar.header("Heatmap Ranges")
spot_min = st.sidebar.number_input("Minimum Spot Price", value=spot_price*0.8)
spot_max = st.sidebar.number_input("Maximum Spot Price", value=spot_price*1.2)
vol_min = st.sidebar.slider("Minimum Volatility", 0.01, 1.0, value=float(volatility*0.5))
vol_max = st.sidebar.slider("Maximum Volatility", 0.01, 1.0, value=float(volatility*1.5))

spot_range = np.linspace(spot_min, spot_max, 10)
vol_range = np.linspace(vol_min, vol_max, 10)

st.title("📈 Black-Scholes Options Pricing Simulator")

# Calculate Call/Put Prices
bs = BlackScholesModel(spot_price, strike_price, time_to_expiry, volatility, risk_free_rate)
call_price, put_price = bs.price()

st.subheader("Calculated Option Prices")
st.metric(label="Call Option Price", value=f"${call_price:.2f}", delta=None)
st.metric(label="Put Option Price", value=f"${put_price:.2f}", delta=None)

# Function to create Heatmaps
def generate_heatmaps(spot_range, vol_range, T, K, r):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))

    for i, sigma in enumerate(vol_range):
        for j, S in enumerate(spot_range):
            model = BlackScholesModel(S, K, T, sigma, r)
            call, put = model.price()
            call_prices[i, j] = call
            put_prices[i, j] = put

    return call_prices, put_prices

# Generate the heatmaps
call_matrix, put_matrix = generate_heatmaps(spot_range, vol_range, time_to_expiry, strike_price, risk_free_rate)

# Plotting
st.markdown("---")
st.subheader("Heatmaps of Option Prices")

col1, col2 = st.columns(2)

with col1:
    st.write("### Call Price Heatmap")
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_matrix, annot=True, fmt=".2f", xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2),
                cmap="RdYlGn", cbar_kws={'label': 'Price'})
    ax1.set_xlabel('Spot Price')
    ax1.set_ylabel('Volatility')
    ax1.set_title('Call Option Price Heatmap')
    st.pyplot(fig1)

with col2:
    st.write("### Put Price Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_matrix, annot=True, fmt=".2f", xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2),
                cmap="RdYlGn", cbar_kws={'label': 'Price'})
    ax2.set_xlabel('Spot Price')
    ax2.set_ylabel('Volatility')
    ax2.set_title('Put Option Price Heatmap')
    st.pyplot(fig2)
