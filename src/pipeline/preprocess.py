from pathlib import Path
import argparse, os
from dotenv import load_dotenv
import pandas as pd
from omegaconf import OmegaConf
from src.features.correlations import compute_correlations

load_dotenv()                       # pulls DATA_ROOT, NUM_THREADS, etc.

def load_config(name: str):
    cfg_path = Path("data/catalog") / f"{name}.yaml"
    return OmegaConf.load(cfg_path)

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--dataset", default="gse40279",
                    help="name of YAML file in data/catalog/ (without .yaml)")
    pa.add_argument("--outdir",  default="data/interim")
    args = pa.parse_args()

    cfg   = load_config(args.dataset)
    betas = pd.read_csv(cfg.betas, index_col=0)
    meta  = pd.read_csv(cfg.meta,  index_col=0)
    ages  = meta.loc[betas.index, cfg.age_column]

    r = compute_correlations(betas, ages)

    out = Path(args.outdir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "correlation.csv")
    print(f"✓ saved {len(r):,} correlations → {out/'correlation.csv'}")

if __name__ == "__main__":
    main()
