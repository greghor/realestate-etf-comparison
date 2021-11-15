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
sp500 = sp500[sp500["Date"] > "1980-01-01"]
sp500["return"] = sp500["SP500"].pct_change()
sp500.describe()
# %

# %
initial_capital = 100_000
investment_horizon = 30
n_sample = 10000
# %

# %
# ETF
invested_capital = initial_capital
etf_return_mean = 0.098
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
etf_summary = etf_values.T.describe(percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]).T
# %


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
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.grid()
# %


# % Real estate
down_payment = 20/100
closing_cost = 10/100
initial_home_value = invested_capital / (down_payment + closing_cost)
loan = initial_home_value *(1 - down_payment)
yearly_interest_rate = 1.5/100
monthly_interest_rate = (1 + yearly_interest_rate)**(1/12) - 1
loan_term = 25
home_appreciation_mean = 3.5/100
home_appreciation_std = 1/100
inflation_mean = 2/100
inflation_std = 1/100

home_returns = np.random.normal(home_appreciation_mean, home_appreciation_std, [investment_horizon, n_sample])
inflation = np.random.normal(inflation_mean, inflation_std, [investment_horizon, n_sample])
compounded_home_returns = (1 + pd.DataFrame(home_returns)).cumprod()
compounded_inflation_effect = (1 + pd.DataFrame(inflation)).cumprod()

# mortgage
monthly_repayment = -np.pmt(monthly_interest_rate, loan_term * 12, loan)

# income
yearly_rent_rentability = 4/100
monthly_rent = (yearly_rent_rentability * initial_home_value)/12
vacancy_rate = 5/100

# recurring operating expense (annual)
property_tax = 1/100
maintenance_cost = 1/100
management_fee = 0.1/100


yearly_cash_flow =  (12 * monthly_rent * (1-vacancy_rate) * compounded_inflation_effect
                     - 12 * monthly_repayment 
                     - (property_tax + maintenance_cost + management_fee) * compounded_inflation_effect * initial_home_value)

yearly_cash_flow.sum()


home_values = initial_home_value * compounded_home_returns - yearly_cash_flow
home_values.T.describe()
home_summary = home_values.T.describe(percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]).T
# %

# %
ax1.fill_between(home_summary.index, home_summary["10%"], home_summary["90%"], color="blue", alpha=0.2)
ax1.fill_between(home_summary.index, home_summary["25%"], home_summary["75%"], color="blue", alpha=0.3)
ax1.plot(home_summary.index, home_summary["50%"], color="blue")
ax1.plot(home_summary.index, home_summary["mean"], "b--")
ax1.set_yscale("log")
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.grid()
plt.show(block=False)
# %

