#!/usr/bin/env python
"""
CLI for selecting top-k CpG features by correlation or mutual information.
"""
from dotenv import load_dotenv
load_dotenv()  # read .env for DATA_ROOT

import argparse
import os
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.pipeline.run_build_noise import load_betas, load_meta
from src.features.select_features import select_top_k_by_corr, select_top_k_by_mi


def load_cfg(dataset: str):
    cfg_path = Path("data/catalog") / f"{dataset}.yaml"
    cfg = OmegaConf.load(str(cfg_path))
    OmegaConf.resolve(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Select top-k CpGs by correlation or mutual information"
    )
    parser.add_argument("--dataset", default="gse40279")
    parser.add_argument("--method", choices=["corr","mi"], default="corr")
    parser.add_argument("--k", type=int, default=5000)
    parser.add_argument("--in-dir", default="data/interim")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    cfg = load_cfg(args.dataset)

    # Load the full beta matrix (handles CSV or pickle)
    betas = load_betas(Path(os.path.expandvars(cfg.betas)))

    # Load precomputed correlations
    corr_path = Path(os.path.expandvars(args.in_dir)) / cfg.name / "correlation.csv"
    corr = pd.read_csv(corr_path, index_col=0).squeeze()

    # Select features
    if args.method == "corr":
        selected = select_top_k_by_corr(betas, corr, args.k)
    else:
        meta = load_meta(Path(os.path.expandvars(cfg.meta)))
        ages = meta[cfg.age_column]
        selected = select_top_k_by_mi(betas, ages, args.k)

    # Write out
    out_path = Path(args.out_dir) / cfg.name
    out_path.mkdir(parents=True, exist_ok=True)
    fp = out_path / f"selected_{args.k}.parquet"
    selected.to_parquet(str(fp), compression="snappy")
    print(f"✓ Wrote top {args.k} features → {fp}")

if __name__ == "__main__":
    main()
