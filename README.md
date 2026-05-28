# Delta-Hedging Simulation and Rebalancing Frequency Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Finance](https://img.shields.io/badge/Finance-Derivatives-green)
![Status](https://img.shields.io/badge/Status-Educational-orange)

## 📊 Description

Monte Carlo simulation of a **delta-hedging** strategy on a European Call under the Black-Scholes model. The project studies the impact of the **rebalancing frequency** (daily, weekly, monthly, quarterly) on the hedging error, and compares with an unhedged strategy.

## 🎯 Objectives

- Quantify the replication error as a function of the rebalancing frequency
- Illustrate the convergence toward a perfect hedge as the time step decreases
- Compare the P&L distribution with and without delta-hedging

## 📐 Mathematical Model

### Underlying dynamics

Under the risk-neutral measure, the price follows a geometric Brownian motion:

$$dS_t = r S_t \, dt + \sigma S_t \, dW_t$$

Trajectories are simulated in log-price using the Euler scheme:

$$\ln S_{t_{k+1}} = \ln S_{t_k} + \left(r - \tfrac{1}{2}\sigma^2\right) h + \sigma \sqrt{h} \, Z_k, \qquad Z_k \sim \mathcal{N}(0,1)$$

### Hedging portfolio

At each rebalancing date, the stock position is adjusted to the Black-Scholes delta:

$$\Delta_t = \Phi(d_1), \qquad d_1 = \frac{\ln(S_t / K) + (r + \tfrac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau}}$$

The discounted hedging P&L is given by:

$$\text{PnL}_T = \sum_{k=0}^{n-1} \Delta_{t_k} \left( S_{t_{k+1}} - S_{t_k} e^{rh} \right) e^{r(T - t_{k+1})}$$

The terminal portfolio value reads:

$$V_T = C_0 \, e^{rT} + \text{PnL}_T - \max(S_T - K, 0)$$

where $C_0$ is the initial call price. A perfect hedge yields $V_T = 0$.

### Unhedged strategy

$$V_T^{\text{no hedge}} = C_0 \, e^{rT} - \max(S_T - K, 0)$$

## 🔧 Parameters

| Parameter | Value |
|-----------|--------|
| $S_0$ | 100 |
| $K$ | 100 (ATM) |
| $r$ | 5% |
| $\sigma$ | 20% |
| $T$ | 1 year |
| Simulations | 16,384 |

### Tested rebalancing frequencies

- **Daily**: $h = 1/252$
- **Weekly**: $h = 1/52$
- **Monthly**: $h = 1/12$
- **Quarterly**: $h = 1/4$

## 📈 Results

The script generates:

- **Histograms** of the terminal portfolio value for each rebalancing frequency and for the unhedged strategy
- **Descriptive statistics** (mean, standard deviation, quantiles) allowing the hedging error to be quantified

## 🚀 Usage
```bash
python delta_hedging.py
```

## 📦 Dependencies
```bash
pip install numpy pandas matplotlib scipy
```

## 👨‍💻 Author

Alexandre R. - Université Paris Cité