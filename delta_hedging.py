"""Simulation of a delta hedge under Black-Scholes.

We sell a European call and hedge it in delta, rebalanced at constant time
steps. We compare the distribution of the final portfolio (capitalized premium
+ hedging P&L - payoff) for several rebalancing frequencies, along with an
unhedged strategy used as a control.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


# Dataclasses for parameters


@dataclass(frozen=True)
class MarketParams:
    """Market and contract parameters

    Parameters ->

    S0 : float
        Spot price of the underlying at t=0
    K : float
        Strike of the option
    r : float
        Risk-free rate
    sigma : float
        Constant volatility of the underlying
    T : float
        Maturity of the option in years
    """

    S0: float = 100.0
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.20
    T: float = 1.0


@dataclass(frozen=True)
class SimConfig:
    """Numerical configuration of the simulation

    Parameters ->

    N : int
        Number of simulated trajectories
    seed : int
        Random seed for reproducibility
    """

    N: int = 16384
    seed: int = 42


# Black-Scholes analytical pricer


class BlackScholesPricer:
    """Analytical pricer for a European call under Black-Scholes.

    Provides the call price and its delta (sensitivity of the price to the
    spot), both needed by the hedging strategy. The methods accept NumPy
    arrays for spot and residual maturity, which allows calling them on all
    trajectories and all time steps at once.
    """

    def __init__(self, params: MarketParams) -> None:
        self.params = params

    def _d1(self, S, tau):
        """d1 term of the Black-Scholes formula for a spot S and a residual
        maturity tau."""
        p = self.params
        return (np.log(S / p.K) + (p.r + 0.5 * p.sigma**2) * tau) / (
            p.sigma * np.sqrt(tau)
        )

    def call_price(self, S, tau):
        """Price of a European call for a spot S and a residual maturity tau.

        Parameters ->

        S : float or ndarray
            Spot price of the underlying
        tau : float or ndarray
            Residual maturity (time remaining until expiry)

        Returns ->

        float or ndarray
            Call price
        """
        p = self.params
        d1 = self._d1(S, tau)
        d2 = d1 - p.sigma * np.sqrt(tau)
        return S * norm.cdf(d1) - p.K * np.exp(-p.r * tau) * norm.cdf(d2)

    def delta(self, S, tau):
        """Delta of a European call for a spot S and a residual maturity tau.

        The delta is the number of shares to hold in order to hedge the option
        at first order.

        Parameters ->

        S : float or ndarray
            Spot price of the underlying
        tau : float or ndarray
            Residual maturity

        Returns ->

        float or ndarray
            Call delta
        """
        return norm.cdf(self._d1(S, tau))


# Hedging strategies


class DeltaHedger:
    """Simulates the delta hedge of a sold call under Black-Scholes.

    We sell a call and collect its premium, invested at the risk-free rate
    until expiry. At each time step we adjust the stock position so that it
    equals the current delta, with purchases and sales financed by borrowing
    at the risk-free rate. At expiry, the final portfolio equals the
    capitalized premium, plus the cumulative hedging P&L, minus the call
    payoff owed to the buyer. A perfect hedge would yield a final portfolio
    of zero; the residual gap measures the hedging error.

    Parameters ->

    params : MarketParams
        Market and contract parameters
    pricer : BlackScholesPricer
        Analytical pricer providing price and delta
    """

    def __init__(self, params: MarketParams, pricer: BlackScholesPricer) -> None:
        self.params = params
        self.pricer = pricer

    def _simulate_paths(self, Z, h):
        """Build price trajectories from Gaussian draws.

        Advances log S time step by time step with the Black-Scholes
        discretization scheme, then maps back to the price level. Each row is
        a trajectory, each column an instant on the grid.

        Parameters ->

        Z : ndarray of shape (N, n)
            Standard Gaussian draws, one row per trajectory
        h : float
            Time step

        Returns ->

        ndarray of shape (N, n + 1)
            Price trajectories including the initial spot in the first column
        """
        p = self.params
        N, n = Z.shape
        log_inc = (p.r - 0.5 * p.sigma**2) * h + p.sigma * np.sqrt(h) * Z
        log_S = np.empty((N, n + 1))
        log_S[:, 0] = np.log(p.S0)
        log_S[:, 1:] = np.log(p.S0) + np.cumsum(log_inc, axis=1)
        return np.exp(log_S)

    def run(self, Z, h):
        """Simulate the delta hedge and return the final portfolio.

        Rebuilds the trajectories, computes the delta at each step, then
        accumulates the discounted hedging P&L. The final portfolio is the
        capitalized premium plus this P&L minus the call payoff.

        Parameters ->

        Z : ndarray of shape (N, n)
            Standard Gaussian draws, one row per trajectory
        h : float
            Rebalancing time step

        Returns ->

        ndarray of shape (N,)
            Portfolio value at expiry for each trajectory
        """
        p = self.params
        N, n = Z.shape
        tau = p.T - np.arange(n) * h

        S = self._simulate_paths(Z, h)
        delta = self.pricer.delta(S[:, :-1], tau)
        premium = self.pricer.call_price(p.S0, p.T)

        # P&L per step: change in stock price minus the financing cost,
        # everything brought back to expiry by capitalization
        discount = np.exp(p.r * (p.T - np.arange(1, n + 1) * h))
        stock_pnl = S[:, 1:] - S[:, :-1] * np.exp(p.r * h)
        hedging_pnl = np.sum(delta * stock_pnl * discount, axis=1)

        payoff = np.maximum(S[:, -1] - p.K, 0.0)
        return premium * np.exp(p.r * p.T) + hedging_pnl - payoff

    def run_no_hedge(self, Z):
        """Simulate the unhedged strategy (control).

        We collect the premium, invest it at the risk-free rate, and suffer
        the call payoff at expiry without ever adjusting a stock position.

        Parameters ->

        Z : ndarray of shape (N,)
            Standard Gaussian draws, one per trajectory

        Returns ->

        ndarray of shape (N,)
            Portfolio value at expiry for each trajectory
        """
        p = self.params
        premium = self.pricer.call_price(p.S0, p.T)
        S_T = p.S0 * np.exp((p.r - 0.5 * p.sigma**2) * p.T + p.sigma * np.sqrt(p.T) * Z)
        return premium * np.exp(p.r * p.T) - np.maximum(S_T - p.K, 0.0)


# Rebalancing frequency study


@dataclass
class HedgingResults:
    """Container for final portfolios by rebalancing frequency."""

    portfolios: dict[str, np.ndarray] = field(default_factory=dict)


class HedgingStudy:
    """Compares the delta hedge for several rebalancing frequencies.

    Parameters ->

    hedger : DeltaHedger
        Configured hedge simulator
    config : SimConfig
        Numerical configuration
    frequencies : dict[str, float]
        Rebalancing frequencies to compare, in the form
        name -> time step h
    """

    DEFAULT_FREQUENCIES = {
        "Daily": 1 / 252,
        "Weekly": 1 / 52,
        "Monthly": 1 / 12,
        "Quarterly": 1 / 4,
    }

    def __init__(
        self,
        hedger: DeltaHedger,
        config: SimConfig,
        frequencies: dict[str, float] | None = None,
    ) -> None:
        self.hedger = hedger
        self.config = config
        self.frequencies = frequencies or self.DEFAULT_FREQUENCIES

    def run(self) -> HedgingResults:
        """Simulate each rebalancing frequency plus the unhedged case.

        Returns ->

        HedgingResults
            Final portfolios indexed by strategy name
        """
        p = self.hedger.params
        rng = np.random.default_rng(seed=self.config.seed)
        res = HedgingResults()

        for label, h in self.frequencies.items():
            n = int(round(p.T / h))
            Z = rng.standard_normal((self.config.N, n))
            res.portfolios[label] = self.hedger.run(Z, h)

        Z_no_hedge = rng.standard_normal(self.config.N)
        res.portfolios["No hedge"] = self.hedger.run_no_hedge(Z_no_hedge)

        return res

    @staticmethod
    def describe(results: HedgingResults) -> pd.DataFrame:
        """Build a table of descriptive statistics of the portfolios.

        Parameters ->

        results : HedgingResults
            Final portfolios to summarize

        Returns ->

        DataFrame
            Descriptive statistics, one row per strategy
        """
        return pd.DataFrame(
            {
                label: pd.Series(values).describe()
                for label, values in results.portfolios.items()
            }
        ).T

    @staticmethod
    def plot(results: HedgingResults) -> None:
        """Display the histogram of the final portfolio for each strategy."""
        labels = list(results.portfolios)
        fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4))

        for axe, label in zip(axes, labels):
            data = results.portfolios[label]
            axe.hist(data, bins=150, alpha=0.75, density=True)
            axe.axvline(
                np.mean(data),
                color="black",
                linestyle="--",
                label=f"mu={np.mean(data):.3f}\n std={np.std(data):.3f}",
            )
            axe.axvline(0, color="red", linestyle=":")
            axe.set_title(label)
            axe.legend()

        fig.suptitle("Delta hedge")
        plt.tight_layout()
        plt.show()


# Main


def main() -> None:
    params = MarketParams(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0)
    config = SimConfig(N=16384, seed=42)

    pricer = BlackScholesPricer(params)
    hedger = DeltaHedger(params, pricer)
    study = HedgingStudy(hedger, config)

    results = study.run()
    print(study.describe(results).to_string())
    study.plot(results)


if __name__ == "__main__":
    main()
