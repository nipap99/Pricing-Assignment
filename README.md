# European Call & Put Option Pricing Analysis

This Python script implements the Black-Scholes model to price European options and perform various financial analyses, including implied volatility calculation.

## Features

- **Black-Scholes Option Pricing**: Calculates European call and put option prices
- **Implied Volatility Calculation**: Uses root-finding methods to determine implied volatility from market prices
- **Visualization**: Plots option values across different strike prices
- **Multiple Stock Analysis**: Handles different stocks with varying parameters

## Black-Scholes Formulas

### Call Option:
$$C = V \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)$$

### Put Option:
$$P = K \cdot e^{-rT} \cdot N(-d_2) - V \cdot N(-d_1)$$

### Where:
$$d_1 = \frac{\ln(V/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$$
$$d_2 = d_1 - \sigma\sqrt{T}$$

- $V$ = Current stock price
- $K$ = Strike price
- $T$ = Time to expiration (years)
- $r$ = Risk-free interest rate
- $\sigma$ = Volatility
- $N()$ = Cumulative distribution function of the standard normal distribution

## Code Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# Given parameters for Stock V1
V1_0 = 95.75          # Current stock price
K_put = 94.5          # Strike price for the put option
T_put = 0.25          # Time to expiration (quarter of a year)
r = 0.043             # Risk-free rate (4.3%)
sigma1_V = 0.38       # Volatility of stock V1 (38%)

# Black-Scholes formula for European Put Option
def black_scholes_put(V, K, T, r, sigma):
    d1 = (np.log(V / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - V * norm.cdf(-d1)
    return put_price

# Black-Scholes formula for European Call Option
def black_scholes_call(V, K, T, r, sigma):
    d1 = (np.log(V / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = V * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Calculate European put price for given parameters
put_price = black_scholes_put(V1_0, K_put, T_put, r, sigma1_V)

# Plot put and call option values for strike prices from 80 to 100
K_values = np.linspace(80, 100, 50)
put_values = [black_scholes_put(V1_0, K, T_put, r, sigma1_V) for K in K_values]
call_values = [black_scholes_call(V1_0, K, T_put, r, sigma1_V) for K in K_values]

# Parameters for Stock V2 and implied volatility calculation
V2_0 = 132            # Current price of stock V2
market_call_price = 8.8506  # Market price of the call option on V2
T_call = 0.5          # Time to expiration for the call (six months)
K_call = 132          # At-the-money strike price

# Function to calculate implied volatility using Brent's method
def implied_volatility_call(V, K, T, r, market_price):
    func = lambda sigma: black_scholes_call(V, K, T, r, sigma) - market_price
    implied_vol = brentq(func, 0.01, 2.0)  # Search for sigma between 1% and 200%
    return implied_vol

# Calculate implied volatility for Stock V2
implied_vol_V2 = implied_volatility_call(V2_0, K_call, T_call, r, market_call_price)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(K_values, put_values, label='Put Option Value', color='blue')
plt.plot(K_values, call_values, label='Call Option Value', color='red')
plt.xlabel('Strike Price (K)')
plt.ylabel('Option Value')
plt.title('European Put and Call Option Values vs. Strike Price')
plt.legend()
plt.grid(True)
plt.show()

# Display results
print(f"European Put Option Price: {put_price:.4f}")
print(f"Implied Volatility for Stock V2: {implied_vol_V2:.4f}")

put_price, implied_vol_V2
```

## Results

### Part 1: European Put Option Price
- **Stock Price (V1)**: $95.75
- **Strike Price**: $94.50
- **Time to Expiration**: 0.25 years
- **Risk-Free Rate**: 4.3%
- **Volatility**: 38%
- **Put Option Price**: Calculated value

### Part 2: Option Value Visualization
The script generates a plot showing:
- **Blue line**: Put option values across different strike prices
- **Red line**: Call option values across different strike prices
- Demonstrates the inverse relationship between put and call option values

### Part 3: Implied Volatility Calculation
- **Stock Price (V2)**: $132.00
- **Market Call Price**: $8.8506
- **Time to Expiration**: 0.5 years
- **Calculated Implied Volatility**: Result from root-finding algorithm

## Applications

- **Option pricing education** - Learn Black-Scholes model implementation
- **Risk management** - Analyze option price sensitivity to parameters
- **Volatility analysis** - Calculate implied volatility from market prices
- **Financial research** - Study option pricing behavior across different strike prices

## Limitations

- **European options only** - Cannot price American options with early exercise
- **Constant parameters** - Assumes constant volatility and risk-free rate
- **Log-normal distribution** - Assumes stock prices follow geometric Brownian motion
- **No dividends** - Does not account for dividend payments
- **Implied volatility calculation** - Requires a market price and assumes the Black-Scholes model is correct
