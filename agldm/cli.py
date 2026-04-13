from __future__ import annotations

import argparse
from pathlib import Path

from agldm.config import load_experiment_config
from agldm.data.prepare import prepare_cub_data
from agldm.evaluation.pipeline import sample_and_evaluate
from agldm.training.classifier import train_attribute_classifier
from agldm.training.ldm import train_ldm
from agldm.training.vqvae import train_vqvae


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AGLDM research MVP CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare_data", help="Prepare CUB manifest and split metadata.")
    prepare.add_argument("--data-config", required=True, type=Path)
    prepare.add_argument("--force", action="store_true")

    for command, help_text in (
        ("train_vqvae", "Train the VQ-VAE on seen classes."),
        ("train_attr_classifier", "Train the attribute classifier on seen classes."),
        ("train_ldm", "Train the latent diffusion model."),
        ("sample_and_eval", "Sample unseen images and compute metrics."),
    ):
        sub = subparsers.add_parser(command, help=help_text)
        sub.add_argument("--data-config", required=True, type=Path)
        sub.add_argument("--model-config", required=True, type=Path)
        sub.add_argument("--train-config", required=True, type=Path)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare_data":
        data_config = load_experiment_config(args.data_config)["data"]
        prepare_cub_data(data_config, force=args.force)
        return

    config = load_experiment_config(args.data_config, args.model_config, args.train_config)
    dispatch = {
        "train_vqvae": train_vqvae,
        "train_attr_classifier": train_attribute_classifier,
        "train_ldm": train_ldm,
        "sample_and_eval": sample_and_evaluate,
    }
    dispatch[args.command](config)


if __name__ == "__main__":
    main()
