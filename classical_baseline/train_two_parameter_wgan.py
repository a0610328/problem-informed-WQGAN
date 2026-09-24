"""Train the exact two-parameter shared classical WGAN control."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from classical_wgan_common import (
    SEEDS,
    ClassicalWGANTrainer,
    TrainingConfig,
    make_two_parameter_generator,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=4001)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume-epoch",
        type=int,
        help="Resume selected runs from this saved 100-epoch checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    invalid_runs = [run for run in args.runs if run not in range(len(SEEDS))]
    if invalid_runs:
        raise ValueError(f"runs must be selected from 0, 1, 2; got {invalid_runs}")

    output_root = args.output_dir
    if output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = Path(__file__).resolve().parent / (
            f"sweep_two_parameter_btc_daily_2020_2026_{stamp}"
        )
    output_root = output_root.resolve()
    config = TrainingConfig(epochs=args.epochs)

    run_progress = tqdm(
        args.runs,
        desc="two-parameter runs",
        unit="run",
        position=0,
        dynamic_ncols=True,
    )
    for run_id in run_progress:
        run_progress.set_postfix(run=run_id, seed=SEEDS[run_id], refresh=True)
        run_dir = (
            output_root
            / f"two_parameter_window_{config.window_size}_stride_{config.stride}"
            / f"run_{run_id}"
        )
        if run_dir.exists() and args.resume_epoch is None:
            raise FileExistsError(
                f"{run_dir} already exists. Use a new --output-dir or "
                "supply --resume-epoch to continue it."
            )
        trainer = ClassicalWGANTrainer(
            generator_factory=make_two_parameter_generator,
            generator_label="two-parameter-shared-control",
            run_id=run_id,
            seed=SEEDS[run_id],
            run_dir=run_dir,
            config=config,
            resume_epoch=args.resume_epoch,
            progress_position=1,
        )
        trainer.train()


if __name__ == "__main__":
    main()
