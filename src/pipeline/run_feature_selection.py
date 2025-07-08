import os
import argparse
import pandas as pd
from omegaconf import OmegaConf
from src.features.select_features import (
    select_top_k_by_corr,
    select_top_k_by_mi
)

def load_cfg(dataset: str):
    p = f"data/catalog/{dataset}.yaml"
    cfg = OmegaConf.load(p)
    OmegaConf.resolve(cfg)
    return cfg

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="gse40279")
    p.add_argument("--method", choices=["corr","mi"], default="corr")
    p.add_argument("--k",      type=int, default=5000)
    p.add_argument("--in-dir", default="data/interim")
    p.add_argument("--out-dir",default="data/processed")
    args = p.parse_args()

    cfg = load_cfg(args.dataset)

    # load the full beta matrix (or noise, if you prefer)
    betas = pd.read_csv(cfg.betas, index_col=0)

    # load age-correlations
    corr = pd.read_csv(
        os.path.join(args.in_dir, cfg.name, "correlation.csv"),
        index_col=0
    ).squeeze()

    if args.method == "corr":
        selected = select_top_k_by_corr(betas, corr, args.k)
    else:
        meta = pd.read_csv(cfg.meta, index_col=0)
        ages = meta[cfg.age_column]
        selected = select_top_k_by_mi(betas, ages, args.k)

    # write out
    out_sub = os.path.join(args.out_dir, cfg.name)
    os.makedirs(out_sub, exist_ok=True)
    fp = os.path.join(out_sub, f"selected_{args.k}.parquet")
    selected.to_parquet(fp, compression="snappy")
    print(f"✓ Wrote top {args.k} features → {fp}")

if __name__ == "__main__":
    main()
