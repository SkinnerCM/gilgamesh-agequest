#!/usr/bin/env python
import os, argparse
import numpy as np
import pandas as pd
import jax
from flax import linen as nn
from src.models.autoencoder import Autoencoder
from src.pipeline.train_ae import load_selected  # reuse helper

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="gse40279")
    p.add_argument("--k",       type=int, default=5000)
    p.add_argument("--encoder-dims", nargs="+", type=int, default=[1024,512])
    p.add_argument("--latent-dim",  type=int, default=128)
    p.add_argument("--decoder-dims", nargs="+", type=int, default=[512,1024])
    args = p.parse_args()

    # 1) load the data you trained on
    data = load_selected(args.dataset, args.k)
    input_dim = data.shape[1]

    # 2) rebuild the model
    model = Autoencoder(
        encoder_dims=args.encoder_dims,
        latent_dim=args.latent_dim,
        decoder_dims=args.decoder_dims,
        input_dim=input_dim
    )

    # 3) load saved params
    param_file = os.path.join("models", args.dataset, f"ae_{args.k}", "checkpoint.npy")
    params = np.load(param_file, allow_pickle=True).item()

    # 4) encode your data in batches
    def encode_batch(x_batch):
        _, z = model.apply({'params': params}, x_batch)
        return np.array(z)

    latents = []
    bs = 32
    for i in range(0, data.shape[0], bs):
        batch = jax.device_put(data[i:i+bs])
        latents.append(encode_batch(batch))
    latents = np.vstack(latents)

    # 5) save to disk
    out_dir = os.path.join("models", args.dataset, f"ae_{args.k}")
    np.save(os.path.join(out_dir, "latents.npy"), latents)
    print(f"✓ Saved latent codes → {out_dir}/latents.npy")

if __name__ == "__main__":
    main()
