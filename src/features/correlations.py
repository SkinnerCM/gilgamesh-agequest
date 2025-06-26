import pandas as pd
import numpy as np

def compute_correlations(betas: pd.DataFrame, ages: pd.Series) -> pd.Series:
    """
    Return Pearson r for every CpG (column) against age.
    Vectorised: runs in seconds even for 450 k CpGs.
    """
    betas_c   = betas.subtract(betas.mean())
    ages_c    = ages - ages.mean()
    denom     = np.sqrt((betas_c**2).sum()) * np.sqrt((ages_c**2).sum())
    r         = betas_c.T.dot(ages_c) / denom
    return pd.Series(r, index=betas.columns)