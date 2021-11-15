# %
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
# %


# %
sp500 = pd.read_csv("sp500.csv")
sp500["Date"] = pd.to_datetime(sp500["Date"])
sp500 = sp500[sp500["Date"].dt.month==1]
#sp500 = sp500[sp500["Date"] > "1948-01-01"]
sp500["return"] = sp500["SP500"].pct_change()
# %

# %
initial_capital = 10_000
investment_horizon = 25
n_sample = 10000
# %

# %
# ETF
invested_capital = 10_000
etf_return_mean = 0.07
etf_return_std = 0.16
etf_returns = np.random.normal(etf_return_mean, etf_return_std, [investment_horizon, n_sample])
broker_buying_fee = 2.5
buying_tax = 0.12/100 * invested_capital
product_charges = 0.25/100 * pd.Series(np.ones(investment_horizon)) * invested_capital
reccurent_fee = 0
# %


# %
etf_conpounded_interest = (1 + pd.DataFrame(etf_returns)).cumprod()
etf_conpounded_interest.mean(axis=1)
# %

# %
etf_values = ((invested_capital - buying_tax - broker_buying_fee) * etf_conpounded_interest).subtract(product_charges, axis=0)
etf_values.mean(axis=1)
# %


etf_summary = etf_values.T.describe(percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]).T


# %
fig, ax1 = plt.subplots()
ax1.set_ylabel("")
ax1.set_xlabel("")
ax1.set_title("")
ax1.fill_between(etf_summary.index, etf_summary["10%"], etf_summary["90%"], color="gray", alpha=0.2)
ax1.fill_between(etf_summary.index, etf_summary["25%"], etf_summary["75%"], color="gray", alpha=0.3)
ax1.plot(etf_summary.index, etf_summary["50%"], color="black")
ax1.plot(etf_summary.index, etf_summary["mean"], "r--")
ax1.set_yscale("log")
ax1.set_yticks([10_000 + (ii*20_000) for ii in range(5)])
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.grid()
plt.show(block=False)
# %
