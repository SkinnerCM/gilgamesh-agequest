# src/pipeline/build_noise.py

from pathlib import Path
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from omegaconf import OmegaConf
from src.features.build_noise import build_noise
from src.utils.io import list_writer

load_dotenv()  # populate os.environ from .env

def load_cfg(dataset: str):
    """
    Load and resolve the YAML config at data/catalog/{dataset}.yaml
    """
    cfg_path = Path("data/catalog") / f"{dataset}.yaml"
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)
    return cfg

def load_betas(path: Path) -> pd.DataFrame:
    """
    Load the beta-matrix from pickle or CSV.
    Assumes index is already 0…(n-1) matching metadata.
    """
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    else:
        return pd.read_csv(path, index_col=0)

def load_meta(path: Path) -> pd.DataFrame:
    """
    Load metadata from Excel or CSV.
    Assumes the column named cfg.age_column is present.
    """
    if path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, index_col=0)
    else:
        return pd.read_csv(path, index_col=0)

def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Build CpG‐age noise residuals using precomputed correlations"
    )
    p.add_argument(
        "--dataset", default="gse40279",
        help="Name of the dataset YAML (without .yaml) in data/catalog/"
    )
    p.add_argument(
        "--corr_dir", default="data/interim",
        help="Where correlation.csv lives (subfolder per dataset)"
    )
    p.add_argument(
        "--outdir", default="data/interim",
        help="Where to write noise.parquet (subfolder per dataset)"
    )
    args = p.parse_args()

    # 1. Load config
    cfg = load_cfg(args.dataset)

    # 2. Build full paths
    betas_path = Path(os.path.expandvars(cfg.betas))
    meta_path  = Path(os.path.expandvars(cfg.meta))
    corr_path  = Path(args.corr_dir) / cfg.name / "correlation.csv"

    # 3. Load inputs
    r      = pd.read_csv(corr_path, index_col=0).squeeze()
    betas  = load_betas(betas_path)
    meta   = load_meta(meta_path)
    ages   = meta[cfg.age_column]

    # 4. Compute noise residuals
    noise = build_noise(betas, ages, r)

    # 5. Save output
    out_dir = Path(args.outdir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Array
    arr = np.array(noise, dtype=np.float32)
    # np.save(out_dir / "noise.npy", arr)

    list_writer(noise.index, out_dir / "samples")
    list_writer(noise.columns, out_dir / "cpgs")

    # # 2) Row labels (samples)
    # np.save(out_dir / "samples.npy", np.array(noise.index, dtype=object))

    # # 3) Column labels (CpGs)
    # np.save(out_dir / "cpgs.npy",    np.array(noise.columns, dtype=object))


    print(f"✓ Saved noise array for {cfg.name} → {out_dir/'noise.npy'}")


if __name__ == "__main__":
    main()
