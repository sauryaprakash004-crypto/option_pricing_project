"""Implied volatility: find the sigma that makes the Black-Scholes price equal
the observed market price. Uses scipy.optimize.brentq (a bracketing root finder)."""

import numpy as np
from scipy.optimize import brentq

from src.black_scholes import bs_price


def implied_volatility(market_price, option_type, S, K, T, r, low=1e-4, high=5.0):
    """Solve  BS_price(sigma) - market_price = 0  for sigma in [low, high].

    BS price increases with sigma, so there is at most one root. If the market
    price is outside what any sigma in the bracket can produce (e.g. below
    intrinsic value), return NaN instead of raising.
    """

    def objective(sigma):
        return bs_price(option_type, S, K, T, r, sigma) - market_price

    try:
        return brentq(objective, low, high, xtol=1e-10)
    except ValueError:  # no sign change in the bracket -> no valid root
        return np.nan
