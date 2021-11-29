# %
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from numpy.random import normal
import numpy_financial as npf
from utils import initialize_values, compound, plot_interval
# %

#############
# PARAMETERS
############

##### GENERAL
INITIAL_CAPITAL = 100_000
INVESTMENT_HORIZON = 30
N_SAMPLE = 10_000
INFLATION_MEAN = 2/100
INFLATION_STD = 1/100
PLOT_PERCENTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
SP_FIGSIZE = [12.12, 4.]

##### REAL ESTATE 
DOWN_PAYMENT = 20/100
CLOSING_COST = 10/100
YEARLY_INTEREST_RATE = 1.5/100
LOAN_TERM = 25
RE_APPRECIATION_MEAN = 3.5/100
RE_APPRECIATION_STD = 8/100
YEARLY_RENT_RENTABILITY = 4.5/100
VACANCY_RATE = 5/100
PROPERTY_TAX = 0.5/100
RENTAL_INCOME_TAX = 10/100
MAINTENANCE_COST = 1/100
MANAGEMENT_FEE = 0.1/100

##### ETF
ETF_RETURN_MEAN = 0.09
ETF_RETURN_STD = 0.16
BROKER_BUYING_FEE = 2.5
BUYING_TAX = 0/100

#############
# MONTE CARLO SIMULATION
############


##### GENERAL
mc_shapes = [INVESTMENT_HORIZON, N_SAMPLE]
inflation = pd.DataFrame(normal(INFLATION_MEAN, INFLATION_STD, mc_shapes))
inflation.iloc[0, :] = 0


##### REAL ESTATE 
initial_re_value = INITIAL_CAPITAL / (DOWN_PAYMENT + CLOSING_COST)
re_values = initialize_values(initial_re_value, mc_shapes)
re_invested_amounts = initialize_values(INITIAL_CAPITAL, mc_shapes)

loan = initial_re_value *(1 - DOWN_PAYMENT)
monthly_interest_rate = (1 + YEARLY_INTEREST_RATE)**(1/12) - 1

yearly_loan_repayments = pd.DataFrame(np.zeros(mc_shapes))
yearly_loan_repayments.iloc[:LOAN_TERM] = -npf.pmt(monthly_interest_rate, LOAN_TERM * 12, loan) * 12

monthly_pmt = npf.ppmt(monthly_interest_rate, np.arange(LOAN_TERM *12)+1, LOAN_TERM * 12, loan)
yearly_pmt = pd.Series([-np.sum(monthly_pmt[:-ii*12]) if ii <= LOAN_TERM else 0 for ii in range(INVESTMENT_HORIZON)])
yearly_pmt.iloc[0] = loan

yearly_rent = (YEARLY_RENT_RENTABILITY * initial_re_value) * (1 - VACANCY_RATE - RENTAL_INCOME_TAX)
yearly_cost = (PROPERTY_TAX + MAINTENANCE_COST + MANAGEMENT_FEE) * initial_re_value
yearly_cash_flow = ((yearly_rent - yearly_cost)* (1 + inflation).cumprod()) - yearly_loan_repayments

re_returns = pd.DataFrame(normal(RE_APPRECIATION_MEAN, RE_APPRECIATION_STD, mc_shapes))

re_values = compound(re_values, re_returns)

yearly_loan_repayments.cumsum().iloc[::-1]

yearly_cash_flow_summary = yearly_cash_flow.T.describe(percentiles=PLOT_PERCENTILES).T
re_ptf = (re_values
              .add(yearly_cash_flow.where(yearly_cash_flow > 0, other=0).cumsum())
              .sub(yearly_pmt, axis=0)
              )
re_summary = re_ptf.T.describe(percentiles=PLOT_PERCENTILES).T
re_invested_amounts = re_invested_amounts - yearly_cash_flow.where(yearly_cash_flow < 0, other=0).shift().fillna(0)
total_re_invested_amounts_summary = re_invested_amounts.cumsum().T.describe(percentiles=PLOT_PERCENTILES).T
# %


##### ETF
etf_returns = pd.DataFrame(normal(ETF_RETURN_MEAN, ETF_RETURN_STD, mc_shapes))
etf_invested_amounts = pd.DataFrame(np.zeros(mc_shapes))
etf_invested_amounts.iloc[0] = INITIAL_CAPITAL
etf_invested_amounts = etf_invested_amounts - yearly_cash_flow.where(yearly_cash_flow < 0, other=0).shift().fillna(0)
etf_invested_amounts = etf_invested_amounts * (1 - BUYING_TAX) - BROKER_BUYING_FEE
etf_values = compound(etf_invested_amounts, etf_returns)

etf_summary = etf_values.T.describe(percentiles=PLOT_PERCENTILES).T


#############
# COMPARE RESULTS
############

# %

fig, ax1 = plt.subplots()
ax1 = plot_interval(etf_summary, ax1, color="red")
ax1.set_ylabel("")
ax1.set_xlabel("")
ax1.set_title("")
ax1.set_yscale("log")
ax1.grid()
plt.show(block=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SP_FIGSIZE)
ax1 = plot_interval(re_summary, ax1, color="blue")
ax1.title.set_text("RE portfolio value")
ax1.set_xlabel("years")
ax1.set_yscale("log")
ax1.grid()
ax2 = plot_interval(total_re_invested_amounts_summary, ax2, color="grey")
ax2.title.set_text("total invested capital")
ax2.set_xlabel("years")
ax2.grid()
plt.show(block=False)

fig, ax1 = plt.subplots()
ax1 = plot_interval(yearly_cash_flow_summary, ax1, color="grey")
ax1.set_ylabel("cash flow")
ax1.set_xlabel("years")
ax1.set_title("")
ax1.grid()
plt.show(block=False)

fig, ax1 = plt.subplots()
ax1 = plot_interval(re_summary, ax1, color="blue", fills=[("10%", "90%")], alphas=[0.3], lines=["50%"], linestyles=["solid"], legend_prefix="RE ")
ax1 = plot_interval(etf_summary, ax1, color="red", fills=[("10%", "90%")], alphas=[0.3], lines=["50%"], linestyles=["solid"], legend_prefix="ETF ")
ax1.set_yscale("log")
ax1.grid()
plt.show(block=False)

fig, ax1 = plt.subplots()
re_ptf_returns = (re_ptf.iloc[-1]/re_invested_amounts.cumsum().iloc[-1])
etf_ptf_returns = (etf_values.iloc[-1]/re_invested_amounts.cumsum().iloc[-1])
re_ptf_returns.hist(ax=ax1, bins=100, color="b", density=1, alpha=0.3)
etf_ptf_returns.hist(ax=ax1, bins=100, color="r", alpha=0.3, density=1)
ax1.set_xlim([0, 25])
ylims = ax1.get_ylim()
ax1.plot([re_ptf_returns.median()]*2, ylims, color="b", label="RE median")
ax1.plot([re_ptf_returns.mean()]*2, ylims, color="b", linestyle="--", label="RE mean")
ax1.plot([etf_ptf_returns.median()]*2, ylims, color="r", label="ETF median")
ax1.plot([etf_ptf_returns.mean()]*2, ylims, color="r", linestyle="--", label="ETF mean")
ax1.title.set_text("distribution of returns at term")
ax1.set_xlabel("returns")
ax1.grid()
plt.show(block=False)

low_val = 1
high_val = 15
print(f"ETF < {low_val}: {len(etf_ptf_returns[etf_ptf_returns < low_val])/N_SAMPLE}")
print(f"RE < {low_val}: {len(re_ptf_returns[re_ptf_returns < low_val])/N_SAMPLE}")
print(f"ETF > {high_val}: {len(etf_ptf_returns[etf_ptf_returns > high_val])/N_SAMPLE}")
print(f"RE > {high_val}: {len(re_ptf_returns[re_ptf_returns > high_val])/N_SAMPLE}")
# %



len(etf_ptf_returns[etf_ptf_returns < 1])/N_SAMPLE
re_ptf_returns[re_ptf_returns < 1]

etf_ptf_returns.mean()
returns = np.random.normal(0.09, 0.16, (30, 1000))
df = pd.DataFrame(returns)
df = 1  + df
dg = df.cumprod()
dg.T.describe().T



