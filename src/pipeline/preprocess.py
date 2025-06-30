from pathlib import Path
import argparse, os
from dotenv import load_dotenv
from omegaconf import OmegaConf
import pandas as pd
from src.features.correlations import compute_correlations

load_dotenv()                          # loads DATA_ROOT

def load_cfg(dataset):
    cfg = OmegaConf.load(f"data/catalog/{dataset}.yaml")
    OmegaConf.resolve(cfg)             # interpolate ${env:...}
    return cfg

def load_betas(path: Path) -> pd.DataFrame:
    if path.suffix == ".pkl":
        return pd.read_pickle(path)

    elif path.suffix in [".csv", ".tsv"]:
        return pd.read_csv(path, index_col=0)
    else:
        raise ValueError(f"Unsupported betas file type: {path.suffix}")

def load_meta(path: Path) -> pd.DataFrame:
    if path.suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, index_col=0)
    elif path.suffix == ".csv":
        return pd.read_csv(path, index_col=0)
    else:
        raise ValueError(f"Unsupported meta file type: {path.suffix}")

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--dataset", default="gse40279")
    pa.add_argument("--outdir",  default="data/interim")
    args = pa.parse_args()

    cfg = load_cfg(args.dataset)

    
    betas_path = Path(os.path.expandvars(cfg.betas))
    meta_path  = Path(os.path.expandvars(cfg.meta))
    
    betas = load_betas(betas_path)
    meta  = load_meta(meta_path)
    ages  = meta[cfg.age_column]


    r = compute_correlations(betas, ages)

    outdir = Path(args.outdir) / cfg.name
    outdir.mkdir(parents=True, exist_ok=True)
    r.to_csv(outdir / "correlation.csv")
    print(f"✓ saved {len(r):,} correlations → {outdir/'correlation.csv'}")

if __name__ == "__main__":
    main()
