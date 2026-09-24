"""Shared training code for the classical WGAN-GP baselines.

The experimental protocol intentionally mirrors ``project_model_ZZ``.  The
generator is the only model component changed by the two public entry points in
this folder.
"""

from __future__ import annotations

import json
import os
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tqdm.auto import tqdm


tf.keras.backend.set_floatx("float64")

BASELINE_DIR = Path(__file__).resolve().parent
# These local modules make the baseline folder portable. Their implementations
# are the preprocessing and metric routines used by project_model_ZZ, reduced
# to the functions required by these two classical experiments.
import data_handling as dh  # noqa: E402
import stylized as st  # noqa: E402


SEEDS = (2, 42, 100)


@dataclass(frozen=True)
class TrainingConfig:
    window_size: int = 16
    stride: int = 1
    epochs: int = 4001
    batch_size: int = 72
    n_critic: int = 5
    eval_interval: int = 5
    checkpoint_interval: int = 100
    evaluation_samples: int = 1000
    gradient_penalty_weight: float = 10.0
    generator_initial_lr: float = 2e-4
    critic_initial_lr: float = 4e-4
    lr_decay_steps: int = 500
    lr_decay_rate: float = 0.9
    min_generator_lr: float = 5e-6
    min_critic_lr: float = 1e-5
    context_window: int = 5


class MinimumExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Exponential decay with the same hard floor as the QGAN training code."""

    def __init__(self, initial_learning_rate, decay_steps, decay_rate, min_lr):
        super().__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float64)
        self.min_lr = tf.cast(min_lr, tf.float64)
        self.exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True,
        )

    def __call__(self, step):
        return tf.maximum(self.exp_decay(step), self.min_lr)

    def get_config(self):
        return {
            "initial_learning_rate": float(self.initial_learning_rate.numpy()),
            "min_lr": float(self.min_lr.numpy()),
        }


def set_global_determinism(seed: int) -> None:
    """Match the random-seed handling used for the quantum experiments."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def make_mixing_critic(window_size: int) -> tf.keras.Model:
    """The same four-channel critic architecture used by ``project_model_ZZ``."""

    main_input = tf.keras.Input(
        shape=(window_size, 4), dtype=tf.float64, name="quad_channel_input"
    )
    x = layers.Conv1D(
        32,
        4,
        padding="valid",
        kernel_regularizer=regularizers.l2(1e-4),
    )(main_input)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.Conv1D(
        64,
        4,
        padding="valid",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv1D(
        128,
        4,
        padding="valid",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1)(x)
    return tf.keras.Model(main_input, output, name="Mixing_Discriminator")


def make_tcnn_generator(window_size: int, filters: int) -> tf.keras.Model:
    """Compact temporal CNN with ``3*f**2 + 8*f + 1`` trainable parameters."""

    if filters < 1:
        raise ValueError("filters must be positive")
    latent = tf.keras.Input(
        shape=(window_size,), dtype=tf.float64, name="latent_noise"
    )
    x = layers.Reshape((window_size, 1))(latent)
    x = layers.Conv1D(filters, 3, padding="same", activation="tanh")(x)
    x = layers.Conv1D(filters, 3, padding="same", activation="tanh")(x)
    x = layers.Conv1D(1, 3, padding="same", activation="tanh")(x)
    output = layers.Reshape((window_size,), name="generated_window")(x)
    return tf.keras.Model(latent, output, name=f"TCNN_Generator_f{filters}")


class TwoParameterSharedLayer(layers.Layer):
    """Translation-invariant local/nearest-neighbour map with two scalars.

    For each position t, the layer evaluates

        tanh(theta_local * z_t
             + theta_neighbour * (z_(t-1) + z_(t+1)) / 2).

    Circular neighbours parallel the ring interaction in the structured QGAN.
    The number of trainable parameters remains two for any window length.
    """

    def build(self, input_shape):
        self.theta_local = self.add_weight(
            name="theta_local",
            shape=(),
            dtype=tf.float64,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
        )
        self.theta_neighbour = self.add_weight(
            name="theta_neighbour",
            shape=(),
            dtype=tf.float64,
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float64)
        neighbour_average = 0.5 * (
            tf.roll(inputs, shift=1, axis=1) + tf.roll(inputs, shift=-1, axis=1)
        )
        return tf.math.tanh(
            self.theta_local * inputs + self.theta_neighbour * neighbour_average
        )


def make_two_parameter_generator(window_size: int) -> tf.keras.Model:
    latent = tf.keras.Input(
        shape=(window_size,), dtype=tf.float64, name="latent_noise"
    )
    output = TwoParameterSharedLayer(name="two_parameter_shared_map")(latent)
    return tf.keras.Model(latent, output, name="Two_Parameter_Shared_Generator")


class ClassicalWGANTrainer:
    """WGAN-GP trainer whose protocol matches the completed QGAN runs."""

    def __init__(
        self,
        generator_factory: Callable[[int], tf.keras.Model],
        generator_label: str,
        run_id: int,
        seed: int,
        run_dir: Path,
        config: TrainingConfig,
        resume_epoch: int | None = None,
        progress_position: int = 0,
    ):
        self.config = config
        self.generator_label = generator_label
        self.run_id = run_id
        self.seed = seed
        self.run_dir = Path(run_dir)
        self.progress_position = progress_position
        self.weights_dir = self.run_dir / "weights"
        self.plots_dir = self.run_dir / "plots"
        self.metrics_dir = self.run_dir / "metrics"
        self.tensorboard_dir = self.run_dir / "tensorboard_logs"
        for directory in (
            self.weights_dir,
            self.plots_dir,
            self.metrics_dir,
            self.tensorboard_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        set_global_determinism(seed)
        self.log_returns = dh.load_BTC_lr().astype(np.float64)
        self.real_windows = dh.chopchop(
            self.log_returns, config.window_size, config.stride
        )
        self.max_lag = config.window_size // 2 - 1
        self.real_metrics = st.calc_window_respecting_pooled_metrics(
            self.real_windows, self.max_lag
        )

        transformed, self.transform_params = dh.transform(
            self.log_returns, unity_transform=False
        )
        self.scale_factor = float(np.max(np.abs(transformed)))
        transformed = transformed / self.scale_factor
        self.train_windows = dh.chopchop(
            transformed, config.window_size, config.stride
        ).astype(np.float64)

        self.generator = generator_factory(config.window_size)
        self.critic = make_mixing_critic(config.window_size)
        self._setup_optimizers(resume_epoch or 0)
        self.summary_writer = tf.summary.create_file_writer(
            str(self.tensorboard_dir)
        )

        self.loss_gen: list[float] = []
        self.loss_disc: list[float] = []
        self.loss_wass: list[float] = []
        self.loss_abs_acf: list[float] = []
        self.loss_raw_acf: list[float] = []
        self.loss_leverage: list[float] = []
        self.loss_epochs: list[int] = []
        self.lowest_wass = float("inf")
        self.lowest_wass_epoch = 0
        self.start_epoch = 0

        if resume_epoch is not None:
            self._resume(resume_epoch)

        self._write_manifest()

    def _setup_optimizers(self, from_epoch: int) -> None:
        config = self.config
        steps_per_epoch = max(
            1, len(self.log_returns) // config.batch_size // config.n_critic
        )
        previous_steps = from_epoch * steps_per_epoch
        decay_factor = config.lr_decay_rate ** (
            previous_steps // config.lr_decay_steps
        )
        generator_lr = max(
            config.generator_initial_lr * decay_factor,
            config.min_generator_lr,
        )
        critic_lr = max(config.critic_initial_lr, config.min_critic_lr)
        generator_schedule = MinimumExponentialDecay(
            generator_lr,
            config.lr_decay_steps,
            config.lr_decay_rate,
            config.min_generator_lr,
        )
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=generator_schedule, clipnorm=1.0
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=critic_lr, clipnorm=1.0
        )

    def _write_manifest(self) -> None:
        manifest = {
            "generator": self.generator_label,
            "run_id": self.run_id,
            "seed": self.seed,
            "generator_parameter_count": self.generator.count_params(),
            "critic_parameter_count": self.critic.count_params(),
            "training": asdict(self.config),
            "noise_distribution": "Uniform[-pi, pi]^16",
            "preprocessing_module": (
                BASELINE_DIR / "data_handling.py"
            ).relative_to(BASELINE_DIR.parent).as_posix(),
            "metric_module": (
                BASELINE_DIR / "stylized.py"
            ).relative_to(BASELINE_DIR.parent).as_posix(),
            "data_file": (
                BASELINE_DIR.parent
                / "Bitcoin_Data_2020_2026"
                / "btc_daily_2020_2026.csv"
            ).relative_to(BASELINE_DIR.parent).as_posix(),
            "critic_source": "Exact architecture copy of project_model_ZZ/models_pennylane_based.py",
        }
        with (self.run_dir / "experiment_manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle, indent=2)

    def get_noise(self, batch_size: int) -> tf.Tensor:
        return tf.random.uniform(
            [batch_size, self.config.window_size],
            minval=-np.pi,
            maxval=np.pi,
            dtype=tf.float64,
        )

    def to_quad_channel(self, log_returns: tf.Tensor) -> tf.Tensor:
        """Exact raw/absolute/squared/rolling-volatility critic representation."""

        # Preserve the reference implementation's float32 feature transform;
        # Keras then casts these channels to the critic's float64 policy.
        x = tf.cast(log_returns, tf.float32)
        c0 = x
        c1 = tf.abs(x)
        c2 = tf.square(x)
        x_expanded = tf.expand_dims(x, axis=-1)
        first = x_expanded[:, 0:1, :]
        padding = tf.tile(first, [1, self.config.context_window - 1, 1])
        padded = tf.concat([padding, x_expanded], axis=1)
        mean_x = tf.nn.avg_pool1d(
            padded,
            ksize=self.config.context_window,
            strides=1,
            padding="VALID",
        )
        mean_x2 = tf.nn.avg_pool1d(
            tf.square(padded),
            ksize=self.config.context_window,
            strides=1,
            padding="VALID",
        )
        c3 = tf.squeeze(
            tf.sqrt(tf.nn.relu(mean_x2 - tf.square(mean_x)) + 1e-7), axis=-1
        )
        return tf.stack([c0, c1, c2, c3], axis=-1)

    @staticmethod
    def critic_wasserstein_loss(real_output, fake_output):
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    @staticmethod
    def generator_wasserstein_loss(fake_output):
        return -tf.reduce_mean(fake_output)

    def gradient_penalty(self, generated, real):
        alpha = tf.random.uniform(
            [tf.shape(real)[0], 1], 0.0, 1.0, dtype=tf.float64
        )
        interpolated = real + alpha * (generated - real)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            predictions = self.critic(
                self.to_quad_channel(interpolated), training=True
            )
        gradients = tape.gradient(predictions, interpolated)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-8)
        return tf.reduce_mean(tf.square(slopes - 1.0))

    @tf.function
    def train_critic_step(self, noise, real):
        real = tf.cast(real, tf.float64)
        with tf.GradientTape() as tape:
            generated = self.generator(noise, training=False)
            real_output = self.critic(self.to_quad_channel(real), training=True)
            fake_output = self.critic(
                self.to_quad_channel(generated), training=True
            )
            base_loss = self.critic_wasserstein_loss(real_output, fake_output)
            penalty = self.gradient_penalty(generated, real)
            loss = base_loss + self.config.gradient_penalty_weight * penalty
        gradients = tape.gradient(loss, self.critic.trainable_variables)
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients, self.critic.trainable_variables
            )
            if gradient is not None
        ]
        self.critic_optimizer.apply_gradients(pairs)
        return loss

    @tf.function
    def train_generator_step(self, noise):
        with tf.GradientTape() as tape:
            generated = self.generator(noise, training=False)
            fake_output = self.critic(
                self.to_quad_channel(generated), training=True
            )
            loss = self.generator_wasserstein_loss(fake_output)
        gradients = tape.gradient(loss, self.generator.trainable_variables)
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients, self.generator.trainable_variables
            )
            if gradient is not None
        ]
        self.generator_optimizer.apply_gradients(pairs)
        return loss, tf.linalg.global_norm([pair[0] for pair in pairs])

    def inverse_transform(self, generated_scaled: np.ndarray) -> np.ndarray:
        return dh.inverse_transform(
            generated_scaled * self.scale_factor, self.transform_params
        )

    def evaluate(self, epoch: int):
        generated_scaled = self.generator(
            self.get_noise(self.config.evaluation_samples), training=False
        ).numpy()
        generated = self.inverse_transform(generated_scaled)
        wasserstein = float(st.metrics(generated, self.real_windows))
        absolute, raw, leverage = st.calc_window_respecting_pooled_metrics(
            generated, max_lag=self.max_lag
        )
        real_absolute, real_raw, real_leverage = self.real_metrics

        self.loss_epochs.append(epoch)
        self.loss_wass.append(wasserstein)
        self.loss_abs_acf.append(float((absolute[0] - real_absolute[0]) ** 2))
        self.loss_raw_acf.append(float((raw[0] - real_raw[0]) ** 2))
        self.loss_leverage.append(float((leverage[0] - real_leverage[0]) ** 2))

        if wasserstein <= self.lowest_wass:
            self.lowest_wass = wasserstein
            self.lowest_wass_epoch = epoch
            self._save_weight_arrays("lowest_wass")
            np.save(self.run_dir / "generated_windows_best.npy", generated)
        return wasserstein, generated

    def train(self) -> None:
        config = self.config
        dataset = (
            tf.data.Dataset.from_tensor_slices(self.train_windows)
            .shuffle(len(self.train_windows))
            .batch(config.batch_size, drop_remainder=True)
        )
        fixed_noise = self.get_noise(config.batch_size)
        generator_step = 0
        latest_generated = None

        print(
            f"Starting {self.generator_label}, run {self.run_id}, seed {self.seed}; "
            f"generator parameters={self.generator.count_params()}"
        )
        try:
            epoch_progress = tqdm(
                range(self.start_epoch, config.epochs),
                total=config.epochs,
                initial=self.start_epoch,
                desc=f"{self.generator_label} | run {self.run_id}",
                unit="epoch",
                dynamic_ncols=True,
                position=self.progress_position,
                leave=False,
            )
            for epoch in epoch_progress:
                last_critic_loss = np.nan
                last_generator_loss = np.nan
                last_gradient_norm = np.nan
                for batch_index, real_batch in enumerate(dataset):
                    critic_noise = self.get_noise(config.batch_size)
                    last_critic_loss = float(
                        self.train_critic_step(critic_noise, real_batch).numpy()
                    )
                    if (batch_index + 1) % config.n_critic == 0:
                        generator_noise = self.get_noise(config.batch_size)
                        generator_loss, gradient_norm = self.train_generator_step(
                            generator_noise
                        )
                        last_generator_loss = float(generator_loss.numpy())
                        last_gradient_norm = float(gradient_norm.numpy())
                        generator_step += 1

                generated_fixed = self.generator(fixed_noise, training=False)
                real_fixed = tf.convert_to_tensor(
                    self.train_windows[: config.batch_size], dtype=tf.float64
                )
                real_output = self.critic(
                    self.to_quad_channel(real_fixed), training=False
                )
                fake_output = self.critic(
                    self.to_quad_channel(generated_fixed), training=False
                )
                self.loss_gen.append(
                    float(self.generator_wasserstein_loss(fake_output).numpy())
                )
                self.loss_disc.append(
                    float(
                        self.critic_wasserstein_loss(
                            real_output, fake_output
                        ).numpy()
                    )
                )

                if epoch % config.eval_interval == 0:
                    wasserstein, latest_generated = self.evaluate(epoch)
                    epoch_progress.set_postfix(
                        W1=f"{wasserstein:.6g}",
                        best=f"{self.lowest_wass:.6g}",
                        refresh=False,
                    )

                with self.summary_writer.as_default():
                    tf.summary.scalar(
                        "Loss/generator_training", last_generator_loss, step=epoch
                    )
                    tf.summary.scalar(
                        "Loss/critic_training", last_critic_loss, step=epoch
                    )
                    tf.summary.scalar(
                        "Gradients/generator_global_norm",
                        last_gradient_norm,
                        step=generator_step,
                    )
                    if self.loss_wass:
                        tf.summary.scalar(
                            "Evaluation/marginal_wasserstein",
                            self.loss_wass[-1],
                            step=epoch,
                        )

                if epoch % config.checkpoint_interval == 0:
                    if latest_generated is None:
                        _, latest_generated = self.evaluate(epoch)
                    self._save_weight_arrays(f"epoch_{epoch}")
                    self._save_qq_plot(latest_generated, epoch)
                    self.save_metrics()
        except KeyboardInterrupt:
            print("Training interrupted; saving the latest metrics and weights.")
            self._save_weight_arrays("interrupted")
        finally:
            self.save_metrics()
            self.summary_writer.flush()

    def _save_qq_plot(self, generated: np.ndarray, epoch: int) -> None:
        figure = st.QQ_plot(
            generated,
            self.log_returns,
            f"Bitcoin Log-Return Q-Q Comparison at Epoch {epoch}",
            xlabel="Generated Bitcoin log-return quantiles",
            ylabel="Real Bitcoin log-return quantiles",
            limit=[-0.04, 0.04],
            show=False,
        )
        figure.savefig(self.plots_dir / f"QQ_plot_epoch_{epoch}.pdf")
        plt.close(figure)

    def _save_weight_arrays(self, label: str) -> None:
        with (self.weights_dir / f"generator_{label}.pkl").open("wb") as handle:
            pickle.dump(self.generator.get_weights(), handle)
        with (self.weights_dir / f"discriminator_{label}.pkl").open(
            "wb"
        ) as handle:
            pickle.dump(self.critic.get_weights(), handle)

    def _resume(self, epoch: int) -> None:
        with (self.weights_dir / f"generator_epoch_{epoch}.pkl").open(
            "rb"
        ) as handle:
            self.generator.set_weights(pickle.load(handle))
        with (self.weights_dir / f"discriminator_epoch_{epoch}.pkl").open(
            "rb"
        ) as handle:
            self.critic.set_weights(pickle.load(handle))
        self._load_metrics_through(epoch)
        self.start_epoch = epoch + 1

    def _load_metrics_through(self, epoch: int) -> None:
        metric_names = {
            "loss_gen": "loss_gen.txt",
            "loss_disc": "loss_disc.txt",
            "loss_wass": "loss_wass.txt",
            "loss_abs_acf": "loss_acf.txt",
            "loss_raw_acf": "loss_acf_nonabs.txt",
            "loss_leverage": "loss_lev.txt",
            "loss_epochs": "loss_epochs.txt",
        }
        loaded = {}
        for attribute, filename in metric_names.items():
            path = self.metrics_dir / filename
            loaded[attribute] = list(np.atleast_1d(np.loadtxt(path)))
        keep = sum(int(value) <= epoch for value in loaded["loss_epochs"])
        for attribute in ("loss_wass", "loss_abs_acf", "loss_raw_acf", "loss_leverage", "loss_epochs"):
            setattr(self, attribute, loaded[attribute][:keep])
        self.loss_gen = loaded["loss_gen"][: epoch + 1]
        self.loss_disc = loaded["loss_disc"][: epoch + 1]
        if self.loss_wass:
            best_index = int(np.argmin(self.loss_wass))
            self.lowest_wass = float(self.loss_wass[best_index])
            self.lowest_wass_epoch = int(self.loss_epochs[best_index])

    def save_metrics(self) -> None:
        arrays = {
            "loss_gen.txt": self.loss_gen,
            "loss_disc.txt": self.loss_disc,
            "loss_wass.txt": self.loss_wass,
            "loss_epochs.txt": self.loss_epochs,
            "loss_acf.txt": self.loss_abs_acf,
            "loss_acf_nonabs.txt": self.loss_raw_acf,
            "loss_lev.txt": self.loss_leverage,
        }
        for filename, values in arrays.items():
            np.savetxt(self.metrics_dir / filename, np.asarray(values))
        np.savetxt(
            self.metrics_dir / "best_metrics.txt",
            np.asarray([self.lowest_wass, self.lowest_wass_epoch]),
        )
