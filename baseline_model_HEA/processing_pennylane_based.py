import data_handling as dh
import stylized as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from functools import reduce
import pennylane as qml  
import pickle
import random
import pandas as pd
import json

import models_pennylane_based as md 

from tqdm import tqdm
import ray
from ray.util import inspect_serializability
import os
from datetime import datetime as dt
import matplotlib.pyplot as plt
from arch import arch_model

# --- GLOBAL FLOAT64 ENFORCEMENT ---
# Forces Keras models and variables to initialize as float64
tf.keras.backend.set_floatx('float64')

def set_global_determinism(seed=42):
    """
    Locks down all random number generators to ensure 100% reproducibility.
    """
    # 1. Set Python hash seed (ensures dictionary/set order is deterministic)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Set Python's built-in random module
    random.seed(seed)
    
    # 3. Set NumPy random seed (Locks down your GARCH noise and PennyLane states)
    np.random.seed(seed)
    
    # 4. Set TensorFlow random seed (Locks down Keras weight initialization and Gradient Penalty alpha)
    tf.random.set_seed(seed)
    
    # 5. (Optional but recommended) Force TensorFlow to use deterministic GPU operations
    # Note: This can slightly slow down training, but guarantees exact reproduction.
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass # Older TF versions might not have this function, safely ignore
        
    print(f"Global random seed set to {seed}")

# Call it immediately!
#set_global_determinism(42)

class MinimumExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, min_lr, staircase=False):
        super().__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float64)
        self.decay_steps = tf.cast(decay_steps, tf.float64)
        self.decay_rate = tf.cast(decay_rate, tf.float64)
        self.min_lr = tf.cast(min_lr, tf.float64)
        self.staircase = staircase
        
        # Initialize the standard decay
        self.exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=self.staircase
        )

    def __call__(self, step):
        # Calculate standard decay
        decayed_lr = self.exp_decay(step)
        # Return the maximum of the decayed LR or the minimum floor!
        return tf.maximum(decayed_lr, self.min_lr)
    
#@ray.remote
class quantum_GAN(object):
    def __init__(self, chopsize, stride, n_qubits, n_layers, epochs, save=False, run_id=1, path='./', resume_path=None, from_epoch = 0):    
        lr = 1e-4
        self.chopsize = chopsize
        self.stride = stride
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.save = save
        self.run_id = run_id
        # The evaluation horizon follows the model output geometry.
        self.max_lag = self.n_qubits - 1

        
        self.qubits = list(range(n_qubits))        
        self.observables = self.generate_observables()

        self.log_returns = dh.load_BTC_lr()
        self.data_real_2d = dh.chopchop(self.log_returns, chopsize, stride)
        self.metrics_real = st.calc_window_respecting_pooled_metrics(self.data_real_2d, self.max_lag)

        self.transformed_lr, self.transform_params = dh.transform(self.log_returns,unity_transform= False)
        
        # Calculate the maximum absolute value to strictly bound data to [-1, 1]
        self.scale_factor = np.max(np.abs(self.transformed_lr))
        #self.scale_factor = 1
        
        # Compress the target data
        self.transformed_lr = self.transformed_lr / self.scale_factor

        # Noise dim matches layers * qubits for distinct noise mapping
        self.noise_dim = 2 * n_qubits 
        
        self.train_time_series = dh.chopchop(self.transformed_lr, chopsize, stride)
        # Cast input series directly to float64
        self.train_time_series = self.train_time_series.reshape(self.train_time_series.shape[0], self.train_time_series.shape[1]).astype('float64')
        self.BUFFER_SIZE = self.train_time_series.shape[0]
        self.BATCH_SIZE = 72 #self.train_time_series.shape[0] // 10

        self.from_epoch = from_epoch
        self.n_critic = 5
        self.eval_interval = 5
                            
        #self._setup_optimizers(self.from_epoch)
        self._setup_optimizers_exp_decay(self.from_epoch)
        #self._prepare_garch_noise(self.log_returns)

        # Create Models
        self.generator = md.generate_model_policy(self.qubits, self.n_layers, self.chopsize, self.observables)
        self.discriminator = md.make_mixing_discriminator(self.chopsize)
        

        if resume_path is not None:
            print(f"Resuming training! Loading weights from {resume_path}")
            self.load_models_custom(resume_path, self.from_epoch)
        
        self.dimension_image_alpha = [1] * len(self.discriminator.input_shape[1:])
        # Update gradient penalty constant to float64
        self.gradient_penalty_weight = tf.constant(10.0, dtype = tf.float64)
        #self.nb_steps_update_critic = 5
        self.loss_gen,self.loss_disc, self.loss_wass, self.loss_ACF, self.loss_ACF_nonabs, self.loss_leverage, self.loss_epochs = [],[],[],[],[],[],[]
        self.lowest_wass = 1e5
        self.lowest_wass_epoch = 0
        self.benchmarks, self.benchmark_lags = None, None
        self.image_info = dict()
        
        if save:
            self.path = os.path.join(path, f'run_{run_id}')
            self.weight_path = os.path.join(self.path, 'weights')
            self.plot_path = os.path.join(self.path, 'plots')
            self.metrics_path = os.path.join(self.path, 'metrics')

            # --- NEW TENSORBOARD SETUP ---
            self.tb_path = os.path.abspath(os.path.normpath(os.path.join(self.path, 'tensorboard_logs')))
            os.makedirs(self.tb_path, exist_ok=True)
            self.summary_writer = tf.summary.create_file_writer(self.tb_path)
            # -----------------------------

            os.makedirs(self.path, exist_ok=True)
            os.makedirs(self.path, exist_ok=True)
            os.makedirs(self.weight_path, exist_ok=True)
            os.makedirs(self.plot_path, exist_ok=True)
            os.makedirs(self.metrics_path, exist_ok=True)

            if resume_path is not None and self.from_epoch > 0:
                try:        
                    resume_idx = self.from_epoch // self.eval_interval
                    self.loss_gen = list(np.atleast_1d(np.loadtxt(self.metrics_path+'/loss_gen.txt')))[:resume_idx]
                    self.loss_disc = list(np.atleast_1d(np.loadtxt(self.metrics_path+'/loss_disc.txt')))[:resume_idx]
                    self.loss_wass = list(np.atleast_1d(np.loadtxt(self.metrics_path+'/loss_wass.txt')))[:resume_idx]
                    self.loss_epochs = list(np.atleast_1d(np.loadtxt(self.metrics_path+'/loss_epochs.txt')))[:resume_idx]                  
                    self.loss_ACF = list(np.atleast_1d(np.loadtxt(self.metrics_path+'/loss_acf.txt')))[:resume_idx]
                    self.loss_ACF_nonabs = list(np.atleast_1d(np.loadtxt(self.metrics_path+'/loss_acf_nonabs.txt')))[:resume_idx]
                    self.loss_leverage = list(np.atleast_1d(np.loadtxt(self.metrics_path+'/loss_lev.txt')))[:resume_idx]
                    best_metrics = np.loadtxt(self.metrics_path+'/best_metrics.txt')
                    self.lowest_wass = best_metrics[0]
                    self.lowest_wass_epoch = int(best_metrics[1])


                    print(f"Successfully loaded {len(self.loss_gen)} previous metric records into memory.")
                    print(f"Resuming search for best EMD. Current best is {self.lowest_wass:.4f} from epoch {self.lowest_wass_epoch}")

                except OSError:
                    print("Could not find old metric files to load. Starting fresh lists.")

    def _setup_optimizers(self, from_epoch):
        """Configures the optimizers and dynamic learning rate schedules for warm restarts."""
        base_gen_lr = 2e-4  
        disc_initial_lr = 4e-4  

        # Dynamically calculate using class attributes
        total_days = len(self.log_returns)
        num_windows = (total_days - self.chopsize) // self.stride + 1
        
        # Use self.BATCH_SIZE and self.n_critic
        total_batches = num_windows // self.BATCH_SIZE
        steps_per_epoch = max(1, total_batches // self.n_critic) 
        
        total_previous_steps = from_epoch * steps_per_epoch

        # Configure Cosine Decay with Restarts
        first_decay_steps = 1000 
        
        gen_lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=base_gen_lr,
            first_decay_steps=first_decay_steps,
            t_mul=2.0,  
            m_mul=0.9,  
            alpha=0.05  
        )

        if from_epoch > 0:
            current_gen_lr = gen_lr_schedule(total_previous_steps).numpy()
            print(f"Warm Start LR: Generator starting at {current_gen_lr:.2e} (Step {total_previous_steps})")
        
        # 在你的 TensorFlow 优化器中直接加入 clipvalue 或 clipnorm
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr_schedule, clipnorm = 1.0)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_initial_lr, clipnorm=1.0)


    def _setup_optimizers_exp_decay(self, from_epoch, crash_penalty_factor=1.0):
        """Configures the optimizers and dynamic learning rate schedules with a hard floor."""
        base_gen_lr = 2e-4  
        disc_initial_lr = 4e-4 
        decay_steps = 500
        decay_rate = 0.9

        # [NEW] Define your safety floors (e.g., 1e-6 or 5e-6)
        MIN_GEN_LR = 5e-6
        MIN_DISC_LR = 1e-5

        # Dynamically calculate steps based on your actual data
        steps_per_epoch = max(1, len(self.transformed_lr) // self.BATCH_SIZE // self.n_critic)
        total_previous_steps = from_epoch * steps_per_epoch

        # Calculate the decayed learning rate and apply the rollback penalty
        decay_factor = decay_rate ** (total_previous_steps // decay_steps)
        current_gen_lr = base_gen_lr * decay_factor * crash_penalty_factor
        current_disc_lr = disc_initial_lr * crash_penalty_factor

        # [NEW] Ensure that even on a rollback / warm restart, the LR hasn't dipped below the floor
        current_gen_lr = max(current_gen_lr, MIN_GEN_LR)
        current_disc_lr = max(current_disc_lr, MIN_DISC_LR)

        if from_epoch > 0:
            print(f"Warm Start LR: Generator starting dynamically at {current_gen_lr:.2e} (Floor: {MIN_GEN_LR:.2e})")

        # [NEW] Use the Custom Schedule Class 
        gen_lr_schedule = MinimumExponentialDecay(
            initial_learning_rate=current_gen_lr,
            decay_steps=decay_steps, 
            decay_rate=decay_rate,
            min_lr=MIN_GEN_LR,       # <--- The floor is enforced here during training!
            staircase=True
        )

        # Attach the optimizers (Don't forget clipnorm to prevent NaNs!)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr_schedule, clipnorm=1.0)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=current_disc_lr, clipnorm=1.0)

                
    def generate_observables(self):
        """
        Output: [X0, Z0, X1, Z1, X2, Z2, ...]
        Generates 2 * n_qubits observables.
        """
        observables_x = []
        observables_z = []
        for i in range(self.n_qubits):
            observables_x.append(qml.PauliX(i))
            observables_z.append(qml.PauliZ(i))
   
        return observables_x + observables_z

    

    @tf.function
    def wasserstein_loss_critic(self, real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
        """The Wasserstein loss of the discriminator."""
        # Enforce float64 inside the loss calculation
        real_output = tf.cast(real_output, tf.float64)
        fake_output = tf.cast(fake_output, tf.float64)
        
        real_loss = tf.reduce_mean(real_output)
        fake_loss = tf.reduce_mean(fake_output)
        
        return tf.cast(fake_loss - real_loss, tf.float64)

    @tf.function
    def wasserstein_loss_generator(self, fake_output: tf.Tensor) -> tf.Tensor:
        """The Wasserstein loss of the generator."""
        fake_output = tf.cast(fake_output, tf.float64)
        return tf.cast(-tf.reduce_mean(fake_output), tf.float64)
    
    @tf.function
    def gradient_penalty(self, critic:tf.keras.Model, fake_generator_output: tf.Tensor, images: tf.Tensor) -> tf.Tensor:
        """ Penalty from the gradients"""

        fake_generator_output = tf.cast(fake_generator_output, tf.float64)
        images = tf.cast(images, tf.float64)

        # Enforce float64 on the random alpha scalar
        alpha = tf.random.uniform(
                shape=[tf.shape(images)[0], 1], minval=0.0, maxval=1.0, dtype=tf.float64
                )
        
        difference_fake_real = fake_generator_output - images
        interpolation = images + (alpha * difference_fake_real)
        
        with tf.GradientTape() as tape:
            tape.watch(interpolation)
            quad_interpolation = self.to_quad_channel(interpolation)

            # Pass the 2-channel interpolation to the critic
            preds = critic(quad_interpolation, training=True)
            
        gradients = tape.gradient(preds, [interpolation])[0]
        assert gradients is not None
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]) + 1e-8)
        
        # Enforce float64 on the norm target
        norm_target = tf.constant(1.0, dtype=tf.float64)
        gp = tf.reduce_mean((slopes - norm_target) ** 2)

        return tf.cast(gp, tf.float64)

    def _prepare_garch_noise(self, raw_log_returns):
        """
        利用真实数据拟合 EGARCH，生成噪声序列，并将其直接标准化缩放到量子门旋转的角度。
        """
        # 1. 将收益率放大 100 倍，拯救 arch 库优化器，避免 Underflow
        scaled_returns = raw_log_returns * 100.0
        
        # 2. 拟合 EGARCH(1,1,1)
        am = arch_model(scaled_returns, vol='EGARCH', p=1, o=1, q=1, dist='Normal')
        res = am.fit(disp='off')
        
        # 3. 模拟生成海量路径 (此时数据处于放大 100 倍的尺度)
        simulated_data = am.simulate(res.params, len(raw_log_returns))
        garch_noise_pool_raw = simulated_data['data'].values
        
        # 4. 计算旋转角缩放因子：将 1 个标准差映射到 pi/4 弧度
        # 数学上，基于 std 的标准化会自动抵消之前放大的 100 倍，因此无需手动除以 100
        garch_std = np.std(garch_noise_pool_raw)
        self.angle_scale_factor = (np.pi / 4.0) / garch_std
        
        # 5. 应用放大因子，得到最终喂给量子门的池子
        garch_noise_pool_scaled = garch_noise_pool_raw * self.angle_scale_factor
        
        print(f"--- 预处理报告 ---")
        print(f"GARCH 原始标准差 (放大100倍后尺度): {garch_std:.5f}")
        print(f"最终旋转角缩放因子 (lambda): {self.angle_scale_factor:.4f}")
        
        # 6. 切割成窗口
        # 注意：此处假定你使用类似 dh.chopchop 的切窗函数，请根据你的实际情况取消注释
        self.garch_noise_windows = dh.chopchop(garch_noise_pool_scaled, self.chopsize, self.stride)
        self.garch_noise_windows = self.garch_noise_windows.astype('float64')

    @tf.function
    def get_hybrid_noise(self, batch_size):
        """Generates the (n_qubit * n_layers)-dim Hybrid Memory-Innovation noise."""
        # 1. Randomly sample 'batch_size' windows from the 16-dim GARCH memory pool
        idx = tf.random.uniform([batch_size], minval=0, maxval=len(self.garch_noise_windows), dtype=tf.int32)
        garch_memory = tf.gather(self.garch_noise_windows, idx)
        garch_memory = tf.cast(garch_memory, tf.float64) # Ensure float64
        
        # 2. 空间-时间切片
        s_even = garch_memory[:, ::2]  # 偶数天: Day 0, 2, 4... (8 维)
        s_odd = garch_memory[:, 1::2]  # 奇数天: Day 1, 3, 5... (8 维)

        # 3. Generate pure Gaussian noise for random innovation
        gaussian_innovation = tf.random.normal([batch_size, self.n_qubits * (self.n_layers-4)], mean=0.0, stddev=1.0, dtype=tf.float64)
        
        # 3. Concatenate them to form the 40-dim input for the QNN
        return tf.concat([s_even, s_odd, s_even, s_odd, gaussian_innovation], axis=1)

    @tf.function
    def get_noise(self, batch_size):
        """
        Generates 16-dimensional uniform noise for strict Data Re-uploading.
        The bounds are set to [-pi, pi] to act directly as rotation angles.
        """
        # self.noise_dim should be set to 2 * self.n_qubits (e.g., 16) in __init__
        noise = tf.random.uniform(
            shape=[batch_size, self.noise_dim], 
            minval=-np.pi, 
            maxval=np.pi, 
            dtype=tf.float64
        )
        
        return noise


    import tensorflow as tf

    def to_quad_channel(self, log_returns, context_window=5):
        """
        Transforms a 2D tensor of log returns (batch_size, sequence_length) 
        into a 4-channel representation using pure TensorFlow operations 
        to preserve the gradient tape.
        """
        # Ensure input is a float32 tensor
        x = tf.cast(log_returns, tf.float32)
        
        # Channel 0: Raw log returns
        c0 = x
        
        # Channel 1: Absolute returns
        c1 = tf.abs(x)
        
        # Channel 2: Squared returns
        c2 = tf.square(x)
        
        # --- Channel 3: Multi-lag context (Rolling Volatility) ---
        # 1. Add a channel dimension for pooling: shape becomes (batch_size, seq_len, 1)
        x_expanded = tf.expand_dims(x, axis=-1)
        
        # 2. Simulate Pandas .bfill() by repeating the first timestep's value 
        # to pad the beginning of the sequence by (context_window - 1) steps.
        first_elements = x_expanded[:, 0:1, :]
        padding_tensor = tf.tile(first_elements, [1, context_window - 1, 1])
        padded_x = tf.concat([padding_tensor, x_expanded], axis=1)
        
        # 3. Calculate rolling Mean(X) and Mean(X^2) using 1D Average Pooling
        mean_x = tf.nn.avg_pool1d(padded_x, ksize=context_window, strides=1, padding='VALID')
        mean_x2 = tf.nn.avg_pool1d(tf.square(padded_x), ksize=context_window, strides=1, padding='VALID')
        
        # 4. Variance = Mean(X^2) - Mean(X)^2
        # Use ReLU to prevent negative variances caused by tiny floating-point inaccuracies
        variance = tf.nn.relu(mean_x2 - tf.square(mean_x))
        
        # 5. Standard Deviation = sqrt(Variance + epsilon for numerical stability)
        c3 = tf.sqrt(variance + 1e-7)
        
        # Remove the temporary pooling channel dimension
        c3 = tf.squeeze(c3, axis=-1)
        
        # Stack along the final axis to create (batch_size, sequence_length, 4)
        quad_channel_data = tf.stack([c0, c1, c2, c3], axis=-1)
        
        return quad_channel_data
    
    @tf.function
    def train_generator(self, gan_instance:tf.keras.Model, critic:tf.keras.Model, noise: tf.Tensor, real_images: tf.Tensor, step: tf.int64):
        """ Update the generator parameter with its optimizer"""
        # Type safeguard float64
        noise = tf.cast(noise, tf.float64)
        real_images = tf.cast(real_images, tf.float64) # 引入真实数据作为基准
        
        with tf.GradientTape() as tape:
            generated_images = gan_instance(noise, training=False)
            generated_images = tf.cast(generated_images, tf.float64)

            # 1. 转换为 4 通道 Tensor (原生 TF 操作，梯度可完美追踪)
            quad_channel_fake = self.to_quad_channel(generated_images)
            
            # 2. 计算 Critic 视角的 Wasserstein Loss
            fake_output = critic(quad_channel_fake, training=True)
            fake_output = tf.cast(fake_output, tf.float64)
            gen_wass_loss = self.wasserstein_loss_generator(fake_output)
        
        gradients_of_generator = tape.gradient(
            gen_wass_loss, gan_instance.trainable_variables
        )

        # --- NEW TENSORBOARD LOGGING ---
        # 1. Log the global norm of all generator gradients
        grad_norm = tf.linalg.global_norm(gradients_of_generator)
        tf.summary.scalar('Gradients/Generator_Global_Norm', grad_norm, step=step)
        
        # 2. Slice and log the 4 specific (8, 6) dimensions of theta
        for grad, var in zip(gradients_of_generator, gan_instance.trainable_variables):
            if grad is not None:
                # Ensure we are operating on the (4, 8, 6) tensor
                if len(grad.shape) == 3: 
                    
                    # Unroll the loop manually for the 4 slices
                    # (Standard Python loops over fixed ranges work perfectly inside @tf.function)
                    for i in range(grad.shape[0]):
                        # Extract the (8, 6) slice
                        slice_grad = grad[i, :, :] 
                        
                        # Calculate the norm of just this slice
                        slice_norm = tf.norm(slice_grad)
                        
                        # Log it to TensorBoard
                        tf.summary.scalar(f'Gradients_Theta_Slices/Slice_{i}', slice_norm, step=step)
        # -------------------------------

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, gan_instance.trainable_variables)
        )    
    @tf.function
    def train_critic(self, gan_instance:tf.keras.Model, critic:tf.keras.Model, noise: tf.Tensor, images: tf.Tensor):
        """Update the critic parameter with its optimizer"""
        
        # Enforce float64 on all incoming data
        images = tf.cast(images, tf.float64)
        noise = tf.cast(noise, tf.float64)

        with tf.GradientTape() as tape:
            generated_images = gan_instance(noise, training=False)
            generated_images = tf.cast(generated_images, tf.float64)

            quad_channel_images = self.to_quad_channel(images)
            quad_channel_generated_images = self.to_quad_channel(generated_images)
            
            real_output = critic(quad_channel_images, training=True)
            fake_output = critic(quad_channel_generated_images, training=True)

            # Calculate Base Loss
            disc_loss = self.wasserstein_loss_critic(real_output, fake_output)
            
            # Gradient Penalty
            penalty_loss = self.gradient_penalty(critic, generated_images, images)
            
            disc_loss += penalty_loss * self.gradient_penalty_weight
            
            # THE ULTIMATE SAFETY NET: Ensure loss stays float64 before gradient tape applies
            disc_loss = tf.cast(disc_loss, tf.float64)
        
        # The corrected code
        gradients_of_critic = tape.gradient(disc_loss, critic.trainable_variables)
        grads_and_vars = zip(gradients_of_critic, critic.trainable_variables)
        # Filter the pairs safely
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]

        self.discriminator_optimizer.apply_gradients(grads_and_vars)

    def train(self):
        print('Run: ', self.run_id, ' has started')
        count = 0

        # FIXED: Use standard batching with drop_remainder to prevent shape mismatches at the end of an epoch
        train_dataset = tf.data.Dataset.from_tensor_slices(self.train_time_series)\
            .shuffle(self.BUFFER_SIZE)\
            .batch(self.BATCH_SIZE, drop_remainder=True)
            

        # MATCH the shape to BATCH_SIZE so fake_output and real_output dimensions align during loss calculation
        noise_cst = self.get_noise(self.BATCH_SIZE)
        # --- NEW STEP TRACKER ---
        gen_train_step = tf.Variable(0, dtype=tf.int64)

        try:
            # ONLY ONE EPOCH LOOP
            for epoch in range(self.from_epoch, self.epochs):    
                
                for step, discriminator_batch in enumerate(train_dataset):
                    
                    # 1. ALWAYS train the critic on every batch
                    noise = self.get_noise(self.BATCH_SIZE)
                    self.train_critic(self.generator, self.discriminator, noise, discriminator_batch)

                    # 2. ONLY train the generator every n_critic steps
                    if (step + 1) % self.n_critic == 0:
                        noise_gen = self.get_noise(self.BATCH_SIZE)
                        # --- TENSORBOARD CONTEXT WRAPPER ---
                        if self.save:
                            with self.summary_writer.as_default():
                                self.train_generator(self.generator, self.discriminator, noise_gen, discriminator_batch, gen_train_step)
                        else:
                            self.train_generator(self.generator, self.discriminator, noise_gen, discriminator_batch, gen_train_step)

                        # Increment our explicit generator step counter
                        gen_train_step.assign_add(1)
                        # -----------------------------------
                    
                #### LOGGING / TESTING
                generated_images = self.generator(noise_cst, training=False)

                quad_channel_generated_images = self.to_quad_channel(generated_images)
                quad_channel_real_images = self.to_quad_channel(self.train_time_series[:self.BATCH_SIZE])
                
                # FIXED: Slice train_time_series to BATCH_SIZE to prevent Out-Of-Memory errors
                real_output = self.discriminator(quad_channel_real_images, training=False)
                fake_output = self.discriminator(quad_channel_generated_images, training=False)
        
                gen_loss = self.wasserstein_loss_generator(fake_output)
                disc_loss = self.wasserstein_loss_critic(real_output, fake_output)
                
                self.loss_gen.append(gen_loss.numpy())
                self.loss_disc.append(disc_loss.numpy())
                
                if epoch % self.eval_interval == 0:
                    # Generate 1000 samples for stable evaluation using the hybrid noise
                    noise = self.get_noise(1000)
                    
                    generated_data = self.generator(noise, training=False).numpy()
                    
                    # --- NEW INVERSE SCALING LOGIC ---
                    # Stretch the [-1, 1] bounded data back to the Lambert W Gaussian space
                    generated_data = generated_data * self.scale_factor
                    generated_data_transformed = dh.inverse_transform(generated_data, self.transform_params)


                    wass  = st.metrics(generated_data_transformed, self.data_real_2d)
                    mean_abs_gen, mean_nonabs_gen, mean_lev_gen = st.calc_window_respecting_pooled_metrics(
                        generated_data_transformed, max_lag=self.max_lag
                    )
                    mean_abs_real, mean_nonabs_real, mean_lev_real = self.metrics_real

                    self.loss_wass.append(wass)
                    self.loss_epochs.append(epoch)
                    self.loss_ACF.append((mean_abs_gen[0] - mean_abs_real[0])**2)
                    self.loss_ACF_nonabs.append((mean_nonabs_gen[0] - mean_nonabs_real[0])**2)
                    self.loss_leverage.append((mean_lev_gen[0] - mean_lev_real[0])**2)

                    print('Run_id {}, found EMD {} at epoch {}'.format(self.run_id, wass, epoch))
                    
                    # FIXED: Indentation. Only open and save the pickle files IF the model actually improved.
                    if wass <= self.lowest_wass:  
                        count +=1
                        self.lowest_wass = wass
                        self.lowest_wass_epoch = epoch

                        with open(f"{self.weight_path}/lowest_wass_generator.pkl", "wb") as f:
                            pickle.dump(self.generator.get_weights(), f)
                        with open(f"{self.weight_path}/lowest_wass_discriminator.pkl", "wb") as f:
                            pickle.dump(self.discriminator.get_weights(), f)
                    
                if epoch % 100 == 0:
                    fig = st.QQ_plot(generated_data_transformed, self.log_returns, 'Bitcoin Log-Return Q-Q Comparison at Epoch {}'.format(str(epoch)), xlabel = 'Generated Bitcoin log-return quantiles', ylabel = 'Real Bitcoin log-return quantiles', limit = [-0.04,0.04], show = False)
                    fig.savefig(self.plot_path+'/QQ_plot_epoch_{}.pdf'.format(str(epoch )))
                    plt.close(fig)  # Frees the bitmap/canvas memory.
                    self.save_models_custom(epoch)
                    self.save_metrics()

                    gen_mean = tf.reduce_mean(quad_channel_generated_images, axis=[0, 1])
                    gen_std = tf.math.reduce_std(quad_channel_generated_images, axis=[0, 1])
                    
                    real_mean = tf.reduce_mean(quad_channel_real_images, axis=[0, 1])
                    real_std = tf.math.reduce_std(quad_channel_real_images, axis=[0, 1])

                   # --- NEW TENSORBOARD LOGGING FOR CHANNELS ---
                    if self.save:
                        with self.summary_writer.as_default():
                            # Descriptive names updated to match the new non-linear transformations
                            channel_names = [
                                '0_Raw_Log_Returns', 
                                '1_Absolute_Returns', 
                                '2_Squared_Returns', 
                                '3_Rolling_Volatility_Window5'
                            ]
                            
                            for i, name in enumerate(channel_names):
                                # Log Means (Real vs Generated)
                                tf.summary.scalar(f'Channel_{name}/Mean_Real', real_mean[i], step=epoch)
                                tf.summary.scalar(f'Channel_{name}/Mean_Generated', gen_mean[i], step=epoch)
                                
                                # Log Standard Deviations (Real vs Generated)
                                tf.summary.scalar(f'Channel_{name}/Std_Real', real_std[i], step=epoch)
                                tf.summary.scalar(f'Channel_{name}/Std_Generated', gen_std[i], step=epoch)
                    # --------------------------------------------

                            # (Optional) You can keep the JSON logging here if you still want the hard copy
                        self.image_info[str(epoch)] = {
                            "gen_mean": gen_mean.numpy().tolist(),
                            "gen_std": gen_std.numpy().tolist(),
                            "real_mean": real_mean.numpy().tolist(),
                            "real_std": real_std.numpy().tolist()
                        }

        except KeyboardInterrupt: 
            print(f"\nTraining interrupted at epoch {epoch}")
        finally:  
            self.save_metrics()
            with open(os.path.join(self.metrics_path, "image_info.json"), 'w') as f:
                json.dump(self.image_info, f, indent=4) # Added indent=4 for human readability
            
    def save_metrics(self): 
        np.savetxt(self.metrics_path+'/loss_gen.txt', np.array(self.loss_gen))
        np.savetxt(self.metrics_path+'/loss_disc.txt', np.array(self.loss_disc))
        np.savetxt(self.metrics_path+'/loss_wass.txt', np.array(self.loss_wass))
        np.savetxt(self.metrics_path+'/loss_epochs.txt', np.array(self.loss_epochs))
        np.savetxt(self.metrics_path+'/best_metrics.txt', np.array([self.lowest_wass, self.lowest_wass_epoch]))
        np.savetxt(self.metrics_path+'/loss_acf_nonabs.txt', np.array(self.loss_ACF_nonabs))
        np.savetxt(self.metrics_path+'/loss_acf.txt', np.array(self.loss_ACF))
        np.savetxt(self.metrics_path+'/loss_lev.txt', np.array(self.loss_leverage))
            
    def generate_samples(self, weights_path):
        self.generator.load_weights(weights_path)
        
        # Use the hybrid noise generator instead of uniform noise
        noise = self.get_noise(1000)              
        
        generated_data = self.generator(noise, training=False).numpy()
        
        # Restore the scale used to bound the training data before the nonlinear inverse transform.
        generated_data_scaled = generated_data * self.scale_factor
        generated_data_transformed = dh.inverse_transform(generated_data_scaled, self.transform_params)

        fig = st.QQ_plot(generated_data_transformed, self.log_returns, 'Bitcoin Log-Return Q-Q Comparison at Best Epoch', xlabel = 'Generated Bitcoin log-return quantiles', ylabel = 'Real Bitcoin log-return quantiles', limit = [-0.04,0.04], show = False)
        fig.savefig(os.path.join(self.plot_path, 'QQ_best.pdf'))
        plt.close(fig)
        
        #metric  = st.metrics(generated_data_transformed, self.log_returns, self.benchmark_lags, self.benchmarks, only_EMD = True)
        metric  = st.metrics(generated_data_transformed, self.log_returns)
        print(f"Final Evaluation EMD: {metric[0]}")
        
        return generated_data_transformed

    def save_models_custom(self, epoch):
        """Bulletproof method to save quantum and classical weights as raw arrays."""
        gen_weights = self.generator.get_weights()
        disc_weights = self.discriminator.get_weights()
        
        with open(f"{self.weight_path}/generator_epoch_{epoch}.pkl", "wb") as f:
            pickle.dump(gen_weights, f)
            
        with open(f"{self.weight_path}/discriminator_epoch_{epoch}.pkl", "wb") as f:
            pickle.dump(disc_weights, f)
        
        print(f"Successfully saved custom weights for epoch {epoch}")

    def load_models_custom(self, resume_path, epoch):
        """Bulletproof method to inject raw array weights back into the models."""
        try:
            with open(f"{resume_path}/weights/generator_epoch_{epoch}.pkl", "rb") as f:
                gen_weights = pickle.load(f)
                self.generator.set_weights(gen_weights)
                
            with open(f"{resume_path}/weights/discriminator_epoch_{epoch}.pkl", "rb") as f:
                disc_weights = pickle.load(f)
                self.discriminator.set_weights(disc_weights)
                
            print(f"Successfully loaded warm-start weights from epoch {epoch}")
        except FileNotFoundError:
            raise ValueError(f"Could not find previous weights at {resume_path} for epoch {epoch}")
    
