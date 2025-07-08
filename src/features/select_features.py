import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

def select_top_k_by_corr(
    df: pd.DataFrame,
    corr: pd.Series,
    k: int
) -> pd.DataFrame:
    """
    Keep the k CpGs whose |correlation with age| is largest.
    """
    top = corr.abs().nlargest(k).index
    return df.loc[:, top]

def select_top_k_by_mi(
    df: pd.DataFrame,
    ages: pd.Series,
    k: int,
    random_state: int = 0
) -> pd.DataFrame:
    """
    Compute mutual info between each CpG (columns of df) and age,
    then keep the top k features.
    """
    mi = mutual_info_regression(
        df.values,
        ages.values,
        random_state=random_state
    )
    mi_s = pd.Series(mi, index=df.columns)
    top = mi_s.nlargest(k).index
    return df.loc[:, top]