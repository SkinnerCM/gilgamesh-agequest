import pandas as pd
import numpy as np

def compute_correlations(betas: pd.DataFrame, ages: pd.Series) -> pd.Series:
    """
    Compute per-CpG Pearson r against age using raw NumPy arrays
    to avoid any label-alignment issues.
    """
    # Extract raw arrays
    B = betas.values.astype(np.float64)    # shape (n_samples, n_cpgs)
    a = ages.values.astype(np.float64)     # shape (n_samples,)

    # Center each column and the age vector
    Bc = B - B.mean(axis=0, keepdims=True)
    ac = a - a.mean()

    # Numerator: dot each CpG-column (length n_samples) with the age vector
    num = Bc.T.dot(ac)                     # shape (n_cpgs,)

    # Denominator: std per CpG * std of age
    std_B = np.sqrt((Bc**2).sum(axis=0))   # shape (n_cpgs,)
    std_a = np.sqrt((ac**2).sum())         # scalar
    denom = std_B * std_a                  # shape (n_cpgs,)

    # Final Pearson r for each CpG
    r = num / denom                        # shape (n_cpgs,)
    return pd.Series(r, index=betas.columns)