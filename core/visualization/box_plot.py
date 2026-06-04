import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

from typing import List, Dict, Tuple, Optional, Any
from itertools import combinations
from matplotlib.lines import Line2D
from core.statistics.statistical_analysis import wilcoxon_rank_sum_test


def plot_covariate_distributions(
    df: pd.DataFrame,
    covariates_to_plot: Dict[str, str],
    predicted_variable: str,
    adjust_p_values: bool = False,
    pairs: Optional[List[Tuple[Any, Any]]] = None,
    label_mapping: Dict[Any, str] = {True: "yes", False: "no"},
    custom_palette: Dict[Any, str] = {True: "tab:blue", False: "tab:red"},
    xlabel: str = "",
    ylabel: str = "",
    fontsize: int = 7,
    fig_width: int = 6,
    fig_height: int = 4,
    num_columns: int = 4,
    point_size: int = 3,
    show_trend_line: bool = False,
    save_path: Optional[str] = None
) -> None:
    """
    Plot boxplots and stripplots of covariate distributions across classes of a predicted variable,
    with annotated p-values from Wilcoxon rank-sum tests.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        covariates_to_plot (Dict[str, str]): Mapping of column names (keys) to plot titles (values).
        predicted_variable (str): Column in `df` representing the predicted/grouping variable.
        adjust_p_values (bool): Whether to apply FDR correction to p-values.
        pairs (List[Tuple[Any, Any]], optional): List of value pairs to compare in predicted variable.
        label_mapping (Dict[Any, str]): Maps raw category values to display labels on the x-axis.
        custom_palette (Dict[Any, str]): Color mapping for each category.
        xlabel (str): Shared label for x-axis.
        ylabel (str): Shared label for y-axis.
        fontsize (int): Base font size for text.
        num_columns (int): Number of columns in subplot grid.
        fig_width (float): Width of figure (inches).
        fig_height (float): Height of figure (inches).
        show_trend_line (bool): Whether to show dashed line connecting median values.
        save_path (str): Path to save the figure.

    Returns:
        None. Displays a matplotlib figure.
    """
    covariate_names = list(covariates_to_plot.keys())
    num_variables = len(covariate_names)
    num_rows = math.ceil(num_variables / num_columns)
    categories = list(label_mapping.keys())

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(fig_width, fig_height))
    axs = axs.flatten() if isinstance(axs, np.ndarray) else np.array([axs])

    if not pairs:
        pairs = list(combinations(categories, 2))

    results_by_pair = {}
    for cat1, cat2 in pairs:
        # Filter to the two categories being compared
        pair_df = df[df[predicted_variable].isin([cat1, cat2])].copy()
        pair_df[predicted_variable] = pair_df[predicted_variable] == cat1  # normalize to boolean

        # Run the test
        pair_result = wilcoxon_rank_sum_test(
            data=pair_df,
            predicted_variable=predicted_variable,
            variables=covariate_names,
            keep=True,
            max_p_value=np.inf,
            adjust_p_values=adjust_p_values,
        )
        results_by_pair[(cat1, cat2)] = pair_result    

    for i, (var, title) in enumerate(covariates_to_plot.items()):
        ax = axs[i]
        x_ticks = categories

        sns.boxplot(
            data=df,
            x=predicted_variable, y=var, hue=predicted_variable,
            palette=custom_palette, gap=0.02, whis=0, showfliers=False,
            order=x_ticks, ax=ax, boxprops=dict(alpha=0.9), medianprops=dict(linewidth=2),
        )
        sns.stripplot(
            data=df,
            x=predicted_variable, y=var, color=".3", size=point_size,
            order=x_ticks, jitter=True, ax=ax
        )

        # P-value legends for > 2 categories
        if len(categories) > 2:
            # Add p-value annotations
            y_min, y_max = df[var].min(), df[var].quantile(0.98)
            y_height = y_max - y_min
            j = 0
    
            for cat1, cat2 in pairs:
                result_df = results_by_pair[(cat1, cat2)]
                row = result_df[result_df['covariate'] == var]
    
                if not row.empty:
                    p = row['p-value'].values[0]
                    if p <= 0.05:
                        x1, x2 = categories.index(cat1), categories.index(cat2)
                        y = y_min + y_height * (1.05 + j / 5)
                        ax.plot([x1, x2], [y, y], lw=1, color='black')
                        p_label = r"$\it{P}$ < 0.001" if p < 0.001 else fr"$\it{{P}}$ = {p:.3f}"
                        ax.text((x1 + x2) / 2, y * 1.01, p_label, ha='center', va='bottom', fontsize=fontsize - 1)
                        j += 1

        # Optional trend line
        if show_trend_line:
            try:
                medians = df.groupby(predicted_variable)[var].median().reindex(categories)
                ax.plot(np.arange(len(medians)), medians.values, linestyle=(0, (5, 1)), linewidth=1, color='black')
            except Exception:
                pass

        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.1)
    
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([label_mapping[k] for k in x_ticks], fontsize=fontsize - 1)
        ax.set_xlabel(xlabel, fontsize=fontsize-1)
        ax.set_ylabel(ylabel, fontsize=fontsize-2)

        ax.set_axisbelow(True)
        ax.tick_params(axis='y', labelsize=fontsize-1)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_title(title, fontsize=fontsize)

        # Legend for two-category case
        if len(categories) == 2:
            row = results_by_pair[(categories[0], categories[1])]
            p_val_row = row[row['covariate'] == var]
            if not p_val_row.empty:
                p_val = p_val_row['p-value'].values[0]
                label = r"$\it{P}$ < 0.001" if p_val < 0.001 else fr"$\it{{P}}$ = {p_val:.3f}"
                ax.legend(handles=[Line2D([0], [0], color='white', label=label)],
                          prop={'size': fontsize - 1}, loc='upper center', bbox_to_anchor=(0.4, 1.05), borderaxespad=0, frameon=False)
        
    # Hide extra subplots
    for ax in axs[num_variables:]:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.show()