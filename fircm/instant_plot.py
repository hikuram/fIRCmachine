import os
import textwrap
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt

# add_underglow, adapted from mplcyberpunk (https://github.com/dhaitz/mplcyberpunk)
def add_underglow(ax: Optional[plt.Axes] = None, alpha_underglow: float = 0.1) -> None:
    """Add an 'underglow' effect, i.e. faintly color the area below the line."""
    if not ax:
        ax = plt.gca()
    # Because ax.fill_between changes axis limits, save current x/y limits to restore later:
    xlims, ylims = ax.get_xlim(), ax.get_ylim()
    lines = ax.get_lines()
    for line in lines:
        # Parameters to use from the original line:
        x, y = line.get_data(orig=False)
        color = line.get_c()
        transform = line.get_transform()
        try:
            step_type = line.get_drawstyle().split("-")[1]
        except:
            step_type = None
        ax.fill_between(
            x=x, y1=y, y2=[0] * len(y), color=color,
            step=step_type, alpha=alpha_underglow, transform=transform,
        )
    ax.set(xlim=xlims, ylim=ylims)
    
# Plot
def instant_plot(dataframe, peak_idx, fig_name):
    
    plt.style.use(os.path.dirname(os.path.abspath(__file__))+"/example.mplstyle")
    plt.rcParams['font.family'] = ['Arial']
    colnames = dataframe.select_dtypes('number').columns.values.tolist()
    palette = sns.color_palette("viridis_r",len(colnames))
    sns.set_palette(palette)
    xlabel = "# image"
    n_row = int(len(colnames)/2)
    plt.rcParams['figure.figsize'] = (12, 2*n_row)
    fig, axes = plt.subplots(n_row, 2, sharex=True)
    for n, colname in enumerate(colnames[1:n_row+1]):
        sns.lineplot(dataframe, x=xlabel, y=colname, hue=None, legend=None, ax=axes[n, 0], color=palette[n])
        axes[n, 0].set_ylabel(textwrap.fill(colname,20))
        add_underglow(axes[n, 0])
        for x in peak_idx:
            y = dataframe[colname][x]
            axes[n, 0].text(x, y, '{:.8g}'.format(y), ha='center')
    for n, colname in enumerate(colnames[n_row+1:]):
        sns.lineplot(dataframe, x=xlabel, y=colname, hue=None, legend=None, ax=axes[n, 1], color=palette[n+n_row+1])
        axes[n, 1].set_ylabel(textwrap.fill(colname,20))
        add_underglow(axes[n, 1])
        for x in peak_idx:
            y = dataframe[colname][x]
            axes[n, 1].text(x, y, '{:.8g}'.format(y), ha='center')

    plt.savefig(fig_name)


