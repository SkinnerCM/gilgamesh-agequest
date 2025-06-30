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
    # Standard deviations
    sigma_beta = betas.std(axis=0)
    sigma_age = ages.std()

    # Compute slope and intercept per CpG
    slope = r * (sigma_beta / sigma_age)
    intercept = betas.mean(axis=0) - slope * ages.mean()

    # Predicted beta = intercept + slope * age
    preds = np.outer(ages.values, slope.values) + intercept.values

    # Noise residuals
    noise = betas.values - preds

    return pd.DataFrame(noise, index=betas.index, columns=betas.columns)