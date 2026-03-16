# Simulation de Delta-Hedging et Analyse de la Fréquence de Rebalancement

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Finance](https://img.shields.io/badge/Finance-Derivatives-green)
![Status](https://img.shields.io/badge/Status-Educational-orange)

## 📊 Description

Simulation Monte Carlo d'une stratégie de **delta-hedging** sur un Call Européen dans le cadre du modèle de Black-Scholes. Le projet étudie l'impact de la **fréquence de rebalancement** (journalier, hebdomadaire, mensuel, trimestriel) sur l'erreur de couverture, et compare avec une stratégie sans couverture.

## 🎯 Objectifs

- Quantifier l'erreur de réplication en fonction de la fréquence de rebalancement
- Illustrer la convergence vers une couverture parfaite lorsque le pas de temps diminue
- Comparer la distribution du P&L avec et sans delta-hedging

## 📐 Modèle Mathématique

### Dynamique du sous-jacent

Sous la mesure risque-neutre, le prix suit un mouvement brownien géométrique :

$$dS_t = r S_t \, dt + \sigma S_t \, dW_t$$

Les trajectoires sont simulées en log-prix par le schéma d'Euler :

$$\ln S_{t_{k+1}} = \ln S_{t_k} + \left(r - \tfrac{1}{2}\sigma^2\right) h + \sigma \sqrt{h} \, Z_k, \qquad Z_k \sim \mathcal{N}(0,1)$$

### Portefeuille de couverture

À chaque date de rebalancement, la position en actions est ajustée au delta de Black-Scholes :

$$\Delta_t = \Phi(d_1), \qquad d_1 = \frac{\ln(S_t / K) + (r + \tfrac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau}}$$

Le P&L actualisé de la couverture est donné par :

$$\text{PnL}_T = \sum_{k=0}^{n-1} \Delta_{t_k} \left( S_{t_{k+1}} - S_{t_k} e^{rh} \right) e^{r(T - t_{k+1})}$$

La valeur terminale du portefeuille s'écrit :

$$V_T = C_0 \, e^{rT} + \text{PnL}_T - \max(S_T - K, 0)$$

où $C_0$ est le prix initial du call. Une couverture parfaite donne $V_T = 0$.

### Stratégie sans couverture

$$V_T^{\text{no hedge}} = C_0 \, e^{rT} - \max(S_T - K, 0)$$

## 🔧 Paramètres

| Paramètre | Valeur |
|-----------|--------|
| $S_0$ | 100 |
| $K$ | 100 (ATM) |
| $r$ | 5% |
| $\sigma$ | 20% |
| $T$ | 1 an |
| Simulations | 16 384 |

### Fréquences de rebalancement testées

- **Journalier** : $h = 1/252$
- **Hebdomadaire** : $h = 1/52$
- **Mensuel** : $h = 1/12$
- **Trimestriel** : $h = 1/4$

## 📈 Résultats

Le script génère :

- **Histogrammes** de la valeur terminale du portefeuille pour chaque fréquence de rebalancement et pour la stratégie sans couverture
- **Statistiques descriptives** (moyenne, écart-type, quantiles) permettant de quantifier l'erreur de couverture

## 🚀 Utilisation
```bash
python delta_hedging_frequency.py
```

## 📦 Dépendances
```bash
pip install numpy pandas matplotlib scipy
```

## 👨‍💻 Auteur

Alexandre R. - Université Paris Cité