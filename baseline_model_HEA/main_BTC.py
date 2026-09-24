import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


from processing_pennylane_based import quantum_GAN, set_global_determinism
from datetime import datetime as dt

import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('float64')


if __name__ == "__main__":

    import os
    from pathlib import Path

    
    # Get the directory where main.py is located
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    num_qubit = 8 
    chopsize = 2 * num_qubit #16
    stride = 1
    epochs = 4001 
    seed_list = [2, 42, 100]
 
    resume_path = None 
    from_epoch = 0

    
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    path_master = './sweep_btc_daily_2020_2026_'+timestamp
    # Continue from previous training checkpoint
    # resume_path = os.path.join(script_dir, r"\sweep_btc_daily_2020_2026_20260802_215447\layers_1_qubit_8_stride_1\run_0\weights")
    # from_epoch = 2000
    # path_master = './sweep_btc_daily_2020_2026_20260802_215447'

     
    for run_id in range(3):
        for n_layer in range(1, 5):
            print(f'Start run {run_id} and layer {n_layer}')
            set_global_determinism(seed_list[run_id])
            path_layer = path_master+f'/layers_{n_layer}_qubit_{num_qubit}_stride_{stride}'
            run_folder = os.path.join(path_layer, f'run_{run_id}')
            if os.path.exists(run_folder):
                continue
            GANS = quantum_GAN(chopsize, stride, num_qubit, n_layer, epochs, save = True, run_id = run_id, path = path_layer, resume_path = resume_path, from_epoch = from_epoch)
            trained_GANS = GANS.train() 
    

    
