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
    # 1. Load style (Arial settings are handled in the .mplstyle file)
    style_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example.mplstyle")
    plt.style.use(style_path)
    
    # 2. Extract columns to plot (excluding the first column used for x-axis)
    colnames = dataframe.select_dtypes('number').columns.values.tolist()
    plot_cols = colnames[1:] 
    
    # 3. Set color palette and layout
    palette = sns.color_palette("viridis_r", len(plot_cols))
    sns.set_palette(palette)
    xlabel = "# image"
    
    # Calculate required number of rows (ceiling division to handle odd number of columns)
    n_row = (len(plot_cols) + 1) // 2 
    plt.rcParams['figure.figsize'] = (12, 2 * n_row)
    fig, axes = plt.subplots(n_row, 2, sharex=True)
    
    # 4. Flatten axes to a 1D array for a single unified loop
    axes_flat = axes.flatten() 
    for n, colname in enumerate(plot_cols):
        ax = axes_flat[n]
        
        # Plot with Seaborn on the specific axis
        sns.lineplot(data=dataframe, x=xlabel, y=colname, hue=None, legend=None, ax=ax, color=palette[n])
        ax.set_ylabel(textwrap.fill(colname, 20))
        add_underglow(ax)
        
        # 5. Add text for peak values (using .loc to avoid Pandas warnings)
        if peak_idx is not None:
            for x in peak_idx:
                if x in dataframe.index:
                    y = dataframe.loc[x, colname]
                    ax.text(x, y, '{:.8g}'.format(y), ha='center')

    # 6. Hide any remaining empty subplots if the number of columns is odd
    for empty_ax in axes_flat[len(plot_cols):]:
        empty_ax.set_visible(False)

    # Adjust layout to prevent label overlapping
    #plt.tight_layout() 
    plt.savefig(fig_name)
