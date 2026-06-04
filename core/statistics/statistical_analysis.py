import pandas as pd
import numpy as np
import scipy

from pandas.api.types import is_numeric_dtype as is_pandas_dtype_numeric

from statsmodels.stats.multitest import multipletests
from typing import List, Tuple, Dict, Union, Set, Optional


from typing import Set


def wilcoxon_rank_sum_test(
    data: pd.DataFrame,
    predicted_variable: str,
    variables: Optional[Union[List[str], str]] = None,
    keep: bool = False,
    max_p_value: float = 0.05,
    adjust_p_values: bool = False,
) -> pd.DataFrame:
    """
    Perform the Wilcoxon rank-sum test between a binary target variable and numerical features.

    Parameters:
        data (pd.DataFrame): DataFrame containing the dataset.
        predicted_variable (str): Binary target variable (e.g., bool, str like "yes"/"no", or 0/1).
        variables (Optional[List[str] | str]): Variables to include or exclude (see `keep`).
        keep (bool): Whether to include (`True`) or exclude (`False`) the specified variables.
        max_p_value (float): Max p-value threshold for filtering results.
        adjust_p_values (bool): Apply multiple testing correction using FDR if True.

    Returns:
        pd.DataFrame: Summary table with columns:
                      ['covariate', 'wilcoxon-statistic', 'p-value', 'number of samples']
    """
    if isinstance(variables, str):
        variables = [variables]
    elif variables is None:
        variables = data.drop(columns=predicted_variable).select_dtypes(include=['bool', 'object']).columns.tolist()

    # Select columns to use
    if keep:
        test_data = data[variables + [predicted_variable]].copy()
    else:
        test_data = data.drop(columns=variables).copy()

    # Normalize predicted variable into a boolean mask
    unique_vals = data[predicted_variable].dropna().unique()
    if len(unique_vals) != 2:
        raise ValueError(f"Expected binary variable with 2 unique values, got {len(unique_vals)}: {unique_vals}")

    val_true, val_false = unique_vals[0], unique_vals[1]
    test_data[predicted_variable] = test_data[predicted_variable] == val_true  # Convert to boolean mask

    results = {'covariate': [], 'wilcoxon-statistic': [], 'p-value': [], 'number of samples': []}

    for variable in test_data.columns:
        if variable == predicted_variable or not is_pandas_dtype_numeric(test_data[variable]):
            continue

        temp_data = test_data[[predicted_variable, variable]].dropna()
        if temp_data.empty:
            continue

        try:
            mask = temp_data[predicted_variable].to_numpy(dtype=bool)
            values = temp_data[variable].to_numpy()
            group1 = values[mask]
            group2 = values[~mask]

            if len(group1) == 0 or len(group2) == 0:
                continue

            result = scipy.stats.ranksums(group1, group2)
            results['covariate'].append(variable)
            results['wilcoxon-statistic'].append(result.statistic)
            results['p-value'].append(result.pvalue)
            results['number of samples'].append(len(temp_data))
        except Exception as e:
            print(f"Failed to test variable: {variable} ({e})")

    results_df = pd.DataFrame(results)

    if adjust_p_values and not results_df.empty:
        results_df['p-value'] = multipletests(results_df['p-value'], method='fdr_bh')[1]

    results_df = results_df.sort_values('p-value').round(3)
    return results_df[results_df['p-value'] <= max_p_value]
