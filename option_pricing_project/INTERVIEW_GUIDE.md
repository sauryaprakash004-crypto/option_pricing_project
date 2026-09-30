# Interview Guide

**What is a European option?**
A contract giving the right (not the obligation) to buy (call) or sell (put) an asset at a fixed strike price on one date only: maturity. American options can be exercised any time.

**What is Black-Scholes?**
A formula for the fair price of a European option. Inputs: spot, strike, maturity, risk-free rate, volatility. It assumes constant volatility and log-normal prices.

**What are d1 and d2?**
Intermediate terms in the formula. N(d2) is roughly the risk-neutral probability the option finishes in the money; N(d1) is the delta of a call. d2 = d1 - sigma * sqrt(T).

**What is implied volatility?**
The volatility that, put into Black-Scholes, reproduces the observed market price. It is the market's view of future volatility. I solve for it with `brentq`.

**Why is implied volatility different for different strikes?**
Black-Scholes assumes constant volatility, but real markets price in crash risk and demand for protection, so different strikes get different IVs. (In my project the pattern is built into the synthetic data.)

**What is a volatility smile?**
Plot IV against strike: it often looks like a smile or a skew (higher IV at low strikes). It shows Black-Scholes' constant-volatility assumption doesn't hold exactly.

**What is a volatility surface?**
The smile across several maturities: strike, maturity and implied volatility in one 3D plot. Along the maturity direction it shows the term structure.

**What is Vega?**
How much the option price changes for a change in volatility. It is highest for at-the-money options with longer maturity, and it is why IV solving is well behaved: price rises with volatility.

**Why use an LSTM?**
It is a neural network designed for sequences, and volatility is a time series with memory. In this project it is a small demonstration of deep learning, not a claim that it beats other models.

**What exactly is the LSTM predicting?**
The next volatility value from the previous 20 volatility values. It does not predict prices.

**What are MAE and RMSE?**
MAE is the average absolute error. RMSE is the square root of the average squared error, so it penalises large errors more. Both are in volatility units (0.01 = 1 volatility point).

**What are the limitations of this project?**
Synthetic data; constant-volatility Black-Scholes; a single small LSTM with one train/test split and no baseline comparison; no dividends or transaction costs; not a trading system.
