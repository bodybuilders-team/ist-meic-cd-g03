from matplotlib.container import BarContainer
from matplotlib.pyplot import gca
from matplotlib.axes import Axes
from datetime import datetime
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from utils.config import LINE_COLOR, FILL_COLOR
from utils.dslabs_functions import FONT_TEXT

def set_chart_labels(
        ax: Axes, title: str = "", xlabel: str = "", ylabel: str = ""
) -> Axes:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

def set_chart_xticks(
    xvalues: list[str | int | float | datetime], ax: Axes, percentage: bool = False
) -> Axes:
    if len(xvalues) > 0:
        if percentage:
            ax.set_ylim(0.0, 1.0)

        if isinstance(xvalues[0], datetime):
            locator = AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(
                AutoDateFormatter(locator, defaultfmt="%Y-%m-%d")
            )
        rotation: int = 0
        if not any(not isinstance(x, (int, float)) for x in xvalues):
            ax.set_xlim(left=xvalues[0], right=xvalues[-1])
            ax.set_xticks(xvalues, labels=xvalues)
        else:
            rotation = 45

        ax.tick_params(axis="x", labelrotation=rotation, labelsize="xx-small")

    return ax


def plot_bar_chart(
        xvalues: list,
        yvalues: list,
        ax: Axes = None,  # type: ignore
        title: str = "",
        xlabel:
        str = "",
        ylabel: str = "",
        percentage: bool = False,
) -> Axes:
    if ax is None:
        ax = gca()
    ax = set_chart_labels(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax = set_chart_xticks(xvalues, ax=ax, percentage=percentage)
    values: BarContainer = ax.bar(
        xvalues,
        yvalues,
        label=yvalues,
        edgecolor=LINE_COLOR,
        color=FILL_COLOR,
        tick_label=xvalues,
    )
    format = "%.2f" if percentage else "%.0f"
    ax.bar_label(values, fmt=format, fontproperties=FONT_TEXT)

    return ax
