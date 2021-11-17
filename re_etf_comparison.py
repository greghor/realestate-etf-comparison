# %
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from numpy.random import normal
from numpy_financial import pmt
# %

#############
# PARAMETERS
############

##### GENERAL
INITIAL_CAPITAL = 100_000
INVESTMENT_HORIZON = 30
N_SAMPLE = 100_000
INFLATION_MEAN = 2/100
INFLATION_STD = 1/100
PLOT_PERCENTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

##### REAL ESTATE 
DOWN_PAYMENT = 20/100
CLOSING_COST = 10/100
YEARLY_INTEREST_RATE = 1.5/100
LOAN_TERM = 25
RE_APPRECIATION_MEAN = 4/100
RE_APPRECIATION_STD = 2/100
YEARLY_RENT_RENTABILITY = 4/100
VACANCY_RATE = 5/100
PROPERTY_TAX = 1/100
MAINTENANCE_COST = 1/100
MANAGEMENT_FEE = 0.1/100

##### ETF
ETF_RETURN_MEAN = 0.08
ETF_RETURN_STD = 0.16
BROKER_BUYING_FEE = 2.5
BUYING_TAX = 0.12/100

#############
# MONTE CARLO SIMULATION
############

##### UTILS 
def compound(amounts: np.array, returns: pd.DataFrame) -> pd.DataFrame:
    compounded_values = returns.copy()
    compounded_values.iloc[0, :] = amounts[0]
    for ii in range(1, len(amounts)):
        compounded_values.iloc[ii, :] = (amounts[ii] + 
                                    compounded_values.iloc[ii-1]*returns.iloc[ii-1].add(1))
    return compounded_values

def plot_interval(df, ax, color):
    ax.fill_between(df.index, df["10%"], df["90%"], color=color, alpha=0.2)
    ax.fill_between(df.index, df["25%"], df["75%"], color=color, alpha=0.3)
    ax.plot(df.index, df["50%"], color=color)
    ax.plot(df.index, df["mean"], color=color, linestyle="dashed")
    return ax

##### GENERAL
mc_shapes = [INVESTMENT_HORIZON, N_SAMPLE]
inflation = pd.DataFrame(normal(INFLATION_MEAN, INFLATION_STD, mc_shapes))
inflation.iloc[0, :] = 0


##### REAL ESTATE 
initial_re_value = INITIAL_CAPITAL / (DOWN_PAYMENT + CLOSING_COST)
re_invested_amounts = [initial_re_value] + [0] * (INVESTMENT_HORIZON - 1)
loan = initial_re_value *(1 - DOWN_PAYMENT)
monthly_interest_rate = (1 + YEARLY_INTEREST_RATE)**(1/12) - 1

yearly_loan_repayments = [-pmt(monthly_interest_rate, LOAN_TERM * 12, loan) * 12] * LOAN_TERM + [0] * (INVESTMENT_HORIZON - LOAN_TERM)

yearly_rent = (YEARLY_RENT_RENTABILITY * initial_re_value) * (1 - VACANCY_RATE)

yearly_cost = (PROPERTY_TAX + MAINTENANCE_COST + MANAGEMENT_FEE) * initial_re_value

re_returns = pd.DataFrame(normal(RE_APPRECIATION_MEAN, RE_APPRECIATION_STD, mc_shapes))

re_values = compound(re_invested_amounts, re_returns)

yearly_cash_flow = ((yearly_rent - yearly_cost)* (1 + inflation).cumprod()).sub(yearly_loan_repayments, axis=0)

re_values_summary = (re_values.T.describe(percentiles=PLOT_PERCENTILES).T
yearly_cash_flow_summary = yearly_cash_flow.T.describe(percentiles=PLOT_PERCENTILES).T
# %



fig, ax1 = plt.subplots()
ax1 = plot_interval(re_values_summary, ax1, "red")
ax1.set_yscale("log")
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.grid()
plt.show(block=False)

fig, ax1 = plt.subplots()
ax1 = plot_interval(yearly_cash_flow_summary, ax1, "green")
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.grid()
plt.show(block=False)
# %
# %

