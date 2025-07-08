#!/usr/bin/env python
"""
Training script for the Flax Autoencoder.
"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import wandb
import optax
from flax import linen as nn
from flax.training import train_state

from omegaconf import OmegaConf
from dotenv import load_dotenv
from src.models.autoencoder import Autoencoder

# Helper to load selected data

def load_selected(dataset: str, k: int, method: str = "corr") -> np.ndarray:
    parquet_path = Path("data/processed") / dataset / f"selected_{k}.parquet"
    df = pd.read_parquet(str(parquet_path))
    return df.values.astype(np.float32)


def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones([1, model.encoder_dims[0]]))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def mse_loss(params, apply_fn, batch):
    recon, _ = apply_fn({'params': params}, batch)
    return jnp.mean((batch - recon) ** 2)


def train_epoch(state, data, batch_size, rng):
    num_samples = data.shape[0]
    perms = jax.random.permutation(rng, num_samples)
    perms = perms[: (num_samples // batch_size) * batch_size]
    perms = perms.reshape(-1, batch_size)

    epoch_loss = 0
    for perm in perms:
        batch = data[perm]
        def loss_fn(params):
            return mse_loss(params, state.apply_fn, batch)
        grads = jax.grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        epoch_loss += loss_fn(state.params)
    epoch_loss = epoch_loss / perms.shape[0]
    return state, epoch_loss


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Train Flax autoencoder on selected CpG data.")
    parser.add_argument("--dataset", default="gse40279")
    parser.add_argument("--k", type=int, default=5000)
    parser.add_argument("--method", choices=["corr", "mi"], default="corr")
    parser.add_argument("--encoder-dims", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--decoder-dims", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--project", default="agequest-autoencoder")
    args = parser.parse_args()

    # Initialize W&B
    wandb.init(
        project=args.project,
        config=vars(args)
    )

    # Load data
    data = load_selected(args.dataset, args.k, args.method)
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    # Model
    model = Autoencoder(
        encoder_dims=args.encoder_dims,
        latent_dim=args.latent_dim,
        decoder_dims=args.decoder_dims
    )
    state = create_train_state(init_rng, model, args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss = train_epoch(
            state, data, args.batch_size, input_rng
        )
        wandb.log({"epoch": epoch, "train_loss": float(train_loss)})
        print(f"Epoch {epoch}, Train MSE: {train_loss:.6f}")

    # Save final params
    ckpt_dir = Path("models") / args.dataset / f"ae_{args.k}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    param_file = ckpt_dir / "checkpoint.npy"
    np.save(str(param_file), state.params)
    print(f"✓ Saved model parameters → {param_file}")

if __name__ == "__main__":
    main()
