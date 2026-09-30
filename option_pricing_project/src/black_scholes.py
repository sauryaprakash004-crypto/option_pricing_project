"""Black-Scholes pricing for European options (no dividends).

Symbols used everywhere in this project:
    S     = spot price of the underlying
    K     = strike price
    T     = time to maturity in years
    r     = continuously compounded risk-free rate
    sigma = volatility (annualised, e.g. 0.20 = 20%)
"""

import numpy as np
from scipy.stats import norm


def d1(S, K, T, r, sigma):
    """d1 = [ln(S/K) + (r + sigma^2 / 2) * T] / (sigma * sqrt(T))"""
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma):
    """d2 = d1 - sigma * sqrt(T)"""
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S, K, T, r, sigma):
    """European call: C = S * N(d1) - K * exp(-rT) * N(d2)"""
    D1, D2 = d1(S, K, T, r, sigma), d2(S, K, T, r, sigma)
    return S * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)


def put_price(S, K, T, r, sigma):
    """European put: P = K * exp(-rT) * N(-d2) - S * N(-d1)"""
    D1, D2 = d1(S, K, T, r, sigma), d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-D2) - S * norm.cdf(-D1)


def bs_price(option_type, S, K, T, r, sigma):
    """Price a 'call' or 'put' (convenience wrapper)."""
    if option_type == "call":
        return call_price(S, K, T, r, sigma)
    if option_type == "put":
        return put_price(S, K, T, r, sigma)
    raise ValueError("option_type must be 'call' or 'put'")


# ---------------------------------------------------------------- Greeks ----
def delta(option_type, S, K, T, r, sigma):
    """Delta = change in option price for a 1 unit change in the spot price."""
    D1 = d1(S, K, T, r, sigma)
    return norm.cdf(D1) if option_type == "call" else norm.cdf(D1) - 1.0


def gamma(S, K, T, r, sigma):
    """Gamma = change in delta for a 1 unit change in spot (same for call and put)."""
    D1 = d1(S, K, T, r, sigma)
    return norm.pdf(D1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """Vega = change in option price for a 1.00 (100 percentage point) change in
    volatility (same for call and put). Divide by 100 for 'per 1% vol'."""
    D1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(D1) * np.sqrt(T)
