from matplotlib.axes import Axes
import pandas as pd
from typing import List, Tuple
import numpy as np

def initialize_values(initial_value: float, shapes: List) -> pd.DataFrame:
    """ create df of shape shapes, filled with zeros except for the first line

    Parameters
    ----------
    initial_value : float
    shapes : List

    Returns
    -------
    pd.DataFrame

    """
    df = pd.DataFrame(np.zeros(shapes))
    df.iloc[0] = initial_value
    return df

def compound(amounts: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """compound.

    Parameters
    ----------
    amounts : pd.DataFrame
        amounts
    returns : pd.DataFrame
        returns

    Returns
    -------
    pd.DataFrame

    """
    """ replicate the evolution of a portfolio

    Parameters
    ----------
    amounts : pd.DataFrame
        amount invested on each time period
    returns : pd.DataFrame
        asset performance on each time period

    Returns
    -------
    pd.DataFrame

    """
    compounded_values = returns.copy()
    compounded_values.iloc[0, :] = amounts.iloc[0, :]
    for ii in range(1, len(amounts)):
        compounded_values.iloc[ii, :] = (amounts.iloc[ii] + 
                                    compounded_values.iloc[ii-1]*returns.iloc[ii-1].add(1))
    return compounded_values

def plot_interval(df: pd.DataFrame, ax: Axes, color: str, 
                  fills: List[Tuple[str, str]] = [("10%", "90%"), ("25%", "75%")], 
                  alphas: List[float] = [0.2, 0.5], 
                  lines: List[str] = ["50%", "mean"], 
                  linestyles: List[str] = ["solid", "dashed"], 
                  legend_loc: int = 2,
                  legend_prefix: str = "",
                  offset_xticks: bool = True) -> Axes:
    """ helper function to plot results of MC simulations

    Parameters
    ----------
    df : pd.DataFrame
    ax : Axes
    color : str
    fills : List[Tuple[str, str]]
    alphas : List[float]
    lines : List[str]
    linestyles : List[str]
    legend_loc : int
    legend_prefix : str
    offset_xticks : bool

    Returns
    -------
    Axes

    """

    if offset_xticks:
        x_ticks = df.index + 1
    for ii, fill in enumerate(fills):
        label = "-".join(fill)
        ax.fill_between(x_ticks, df[fill[0]], df[fill[1]], color=color, alpha=alphas[ii], label=legend_prefix + label)
    for ii, line in enumerate(lines):
        ax.plot(x_ticks, df[line], color=color, linestyle=linestyles[ii], label=legend_prefix + line)
    ax.legend(loc=legend_loc)
    return ax
