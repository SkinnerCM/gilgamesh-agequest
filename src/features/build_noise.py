import pandas as pd
import numpy as np

def build_noise(betas: pd.DataFrame, ages: pd.Series, r: pd.Series) -> pd.DataFrame:
    """
    Compute noise residuals for each sample and CpG:
    noise_{s,j} = beta_{s,j} - (intercept_j + slope_j * age_s)
    where:
    slope_j = r_j * (sigma_beta_j / sigma_age)
    intercept_j = mean_beta_j - slope_j * mean_age

    Args:
        betas: DataFrame of shape (n_samples, n_cpgs) with beta values.
        ages: Series of shape (n_samples,) with chronological ages.
        r: Series of shape (n_cpgs,) with Pearson r for each CpG against age.

    Returns:
        DataFrame of shape (n_samples, n_cpgs) of noise residuals.
    """

     # Convert inputs to numpy arrays
    ages_arr = np.array(ages, dtype=np.float32)  # shape (n_samples,)
    r_arr    = np.array(r, dtype=np.float32)     # shape (n_cpgs,)

    # Compute statistics
    sigma_beta = np.array(betas.std(axis=0), dtype=np.float32)  # shape (n_cpgs,)
    sigma_age  = ages_arr.std()
    mean_beta  = np.array(betas.mean(axis=0), dtype=np.float32) # shape (n_cpgs,)
    mean_age   = ages_arr.mean()

    # Compute slope and intercept arrays
    slope_arr     = r_arr * (sigma_beta / sigma_age)
    intercept_arr = mean_beta - slope_arr * mean_age

    # Predicted betas and noise residuals
    preds = np.outer(ages_arr, slope_arr) + intercept_arr  # shape (n_samples, n_cpgs)
    noise = np.array(betas, dtype=np.float32) - preds

    return pd.DataFrame(noise, index=betas.index, columns=betas.columns)