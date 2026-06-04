from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def draw_scatter_plot(
    df: pd.DataFrame,
    x_variable: str, 
    y_variable: str, 
    correlation_method='spearman',
    show_trendline=False,
    show_regression=False,
    show_correlation=True,
    title=None,
    x_label=None,
    y_label=None,
    fontsize=7,
    show_grid=False,
    figsize=(2, 2),
    marker_size: int = 10,
    save_path: str = None,
    **kwargs,
):
    """
    General function to plot a single scatter plot with options to show regression line, trend line,
    and correlation coefficients (Spearman or Pearson). Also includes a flag to show or hide the grid.

    Args:
    - df (DataFrame): The data to plot.
    - x_variable (str): The column name in the DataFrame for the x-axis variable.
    - y_variable (str): The column name in the DataFrame for the y-axis variable.
    - correlation_method (str): The correlation method to use ('spearman' or 'pearson').
    - show_trendline (bool): Whether to show a trend line.
    - show_regression (bool): Whether to show a regression line.
    - show_correlation (bool): Whether to show correlation coefficient.
    - title (str): Title of the plot (optional).
    - x_label (str): Custom label for the x-axis.
    - y_label (str): Custom label for the y-axis.
    - fontsize (int): Font size for labels, titles, etc.
    - show_grid (bool): Whether to show the grid (default True).
    - figsize (tuple): Size of the figure (default is (2, 2)).
    - marker_size (int): Marker size for scatter points.
    - save_path (str): Path to save figure.
    """
    # Drop rows where either x or y have NaN values to keep the data synchronized
    df_clean = df.dropna(subset=[x_variable, y_variable])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(data=df_clean, x=x_variable, y=y_variable, color=".3", alpha=1, s=marker_size, ax=ax, **kwargs)

    # Correlation calculation
    x = df_clean[x_variable].to_numpy()
    y = df_clean[y_variable].to_numpy()

    max_val = max(x.max(), y.max())
    
    if correlation_method == 'spearman':
        result = stats.spearmanr(x, y)
        corr_coef = result.correlation
        p_value = result.pvalue
    elif correlation_method == 'pearson':
        result = stats.pearsonr(x, y)
        corr_coef = result.statistic
        p_value = result.pvalue

    p_value = round(p_value, 3)
    
    if show_correlation:
        if p_value < 0.001:
            label = f"R = {corr_coef:.2f} \n($\\it{{P}}$ < 0.001)"
        else:
            label = f"R = {corr_coef:.2f} \n($\\it{{P}}$ = {p_value:.3f})"
        #ax.legend(title=label, title_fontsize=fontsize - 1, frameon=False)
        ax.legend(handles=[Line2D([0], [0], color='white', label=label)], prop={'size': fontsize - 1}, loc='best',  frameon=False)

    if show_regression:
        sns.regplot(data=df_clean, x=x_variable, y=y_variable, order=1, ci=None, scatter=False, line_kws={'color': 'black','linestyle': (0, (5, 1)), 'linewidth': 1}, ax=ax, **kwargs)

    if show_trendline:

        # Choose a tick interval from predefined "nice" values
        nice_intervals = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        target_ticks = 5
        raw_interval = max_val / target_ticks

        tick_interval = min([i for i in nice_intervals if i >= raw_interval], default=nice_intervals[-1])

        ticks = np.arange(0, max_val + tick_interval, tick_interval)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        # Draw trendline (diagonal)
        ax.plot([0, max_val], [0, max_val], color='black', linestyle=(0, (5, 1)), linewidth=1)

        # Set identical axis limits
        ax.set_xlim(0, max_val*1.1)
        ax.set_ylim(0, max_val*1.1)
        
    if x_label:
        ax.set_xlabel(x_label, fontsize=fontsize-1)
    if y_label:
        ax.set_ylabel(y_label, fontsize=fontsize-1)

    ax.xaxis.label.set_fontsize(fontsize-1)
    ax.yaxis.label.set_fontsize(fontsize-1)

    ax.xaxis.set_tick_params(labelsize=fontsize-1)
    ax.yaxis.set_tick_params(labelsize=fontsize-1)

    if title:
        ax.set_title(title, fontsize=fontsize)
        
    ax.set_axisbelow(True)
    ax.spines[['right', 'top']].set_visible(False)

    # Show grid if show_grid is True
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

    plt.show()