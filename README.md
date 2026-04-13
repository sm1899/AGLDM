# AGLDM

This repository contains a PyTorch implementation of the AGLDM paper:

- `prepare_data`: build a CUB manifest from raw dataset assets and standard zero-shot splits.
- `train_vqvae`: train the VQ-VAE / VQGAN image autoencoder on seen classes.
- `train_attr_classifier`: train the frozen attribute predictor on seen classes.
- `train_ldm`: train the latent diffusion model with text + attribute conditioning.
- `sample_and_eval`: sample unseen classes and compute FID, IS, CLIP, and attribute consistency.

Run the CLI with three config families:

```bash
python -m agldm.cli prepare_data --data-config configs/data/cub.yaml
python -m agldm.cli train_vqvae --data-config configs/data/cub.yaml --model-config configs/model/agldm_cub.yaml --train-config configs/train/base.yaml
```
