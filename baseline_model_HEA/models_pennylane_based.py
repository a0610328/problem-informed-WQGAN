import pennylane as qml
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import regularizers

# --- GLOBAL FLOAT64 ENFORCEMENT ---
# Forces Keras models and variables to initialize as float64
tf.keras.backend.set_floatx('float64')


def one_qubit_rotation(qubit, symbols):
    """
    Returns PennyLane operations that apply a rotation of the bloch sphere 
    about the X, Y and Z axis, specified by the values in `symbols`.
    """
    qml.RX(symbols[0], wires=qubit)
    qml.RY(symbols[1], wires=qubit)
    qml.RZ(symbols[2], wires=qubit)

def entangling_layer(qubits):
    """
    Applies a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    for i in range(len(qubits) - 1):
        qml.CZ(wires=[qubits[i], qubits[i+1]])
    # Circular topology: connect last to first if there are more than 2 qubits
    #if len(qubits) != 2:
    #    qml.CZ(wires=[qubits[0], qubits[-1]])

def generate_circuit(qubits, n_layers, params, inputs):
    """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
    for l in range(n_layers):
        # Variational layer
        for i, q in enumerate(qubits):
            # [FIXED]: Added [:,] to strictly extract the 3 rotation angles 
            one_qubit_rotation(q, params[l, i, :])
            
        entangling_layer(qubits)
        
        # Encoding layer (Re-uploading)
        for i, q in enumerate(qubits):
            # [FIXED]: Added `:,` to slice the batch dimension for Native Broadcasting
            qml.RX(inputs[:, l, i], wires=q)

    # Last variational layer
    for i, q in enumerate(qubits):
        # [FIXED]: Added [:,] to strictly extract the 3 rotation angles 
        one_qubit_rotation(q, params[n_layers, i, :])
            


class ReUploadingPQC(tf.keras.layers.Layer):
    """
    Performs the transformation and data re-uploading using PennyLane.
    Inputs are scaled by trainable lambdas and passed through an activation function.
    """

    def __init__(self, qubits, n_layers, observables, activation="linear", name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)
        self.qubits = qubits
        self.observables = observables
        self.activation = tf.keras.activations.get(activation)

        # Initialize theta weights for variational layers - Cast to float64
        self.theta = self.add_weight(
            shape=(self.n_layers + 1, self.n_qubits, 3),
            initializer=tf.random_uniform_initializer(minval=0.0, maxval=np.pi),
            dtype="float64",
            trainable=True,
            name="thetas"
        )

        # Initialize lambda weights for input scaling - Cast to float64
        self.lmbd = self.add_weight(
            shape=(self.n_layers, self.n_qubits),
            initializer=tf.ones_initializer(),
            dtype="float64",
            trainable=True,
            name="lambdas"
        )

        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(inputs, weights):
            generate_circuit(self.qubits, self.n_layers, weights, inputs)
            
            # [FIXED]: Explicitly return BOTH Z and X measurements (Output size = 2 * n_qubits)
            #z_measurements = [qml.expval(qml.PauliZ(wires=q)) for q in self.qubits]
            #x_measurements = [qml.expval(qml.PauliX(wires=q)) for q in self.qubits]
        
            #return [item for pair in zip(z_measurements, x_measurements) for item in pair]

            # This listens to the processing script!
            return [qml.expval(obs) for obs in self.observables]

        self.qnode = circuit

    def call(self, inputs, **kwargs): 

        #1. Expand dimensions to create a layer axis: [Batch, 1, 16]
        expanded_inputs = tf.expand_dims(inputs, axis=1)
        
        # 2. Tile the EXACT SAME 16 values across all n_layers: [Batch, n_layers, 16]
        reshaped_inputs = tf.tile(expanded_inputs, [1, self.n_layers, 1])
        results = self.qnode(reshaped_inputs, self.theta)
        
        if isinstance(results, (list, tuple)):
            results = tf.stack(results, axis=1)

        results = tf.math.real(results)
        results = tf.cast(results, tf.float64) # Enforce float64 cast
        
        # [FIXED]: The "Ghost Tensor" anchor. Output size is now 2 * length of qubits.
        results = tf.ensure_shape(results, [None, 2 * len(self.qubits)])

        return results
    
    def compute_output_shape(self, input_shape):
        # [FIXED]: Match the new double output size
        return tf.TensorShape([input_shape[0], 2 * len(self.qubits)])
    

def generate_model_policy(qubits, n_layers, n_actions, observables):
    """Generates a Keras model for a data re-uploading PQC policy."""

    input_tensor = tf.keras.Input(shape=(2 * len(qubits), ), dtype=tf.float64, name='input')
    
    # Pass inputs to the ReUploading PQC
    re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables)(input_tensor)
    
    model = tf.keras.Model(inputs=[input_tensor], outputs=re_uploading_pqc)

    return model


def make_mixing_discriminator(chopsize):
    # Input shape is (chopsize, 4 channels)
    main_input = tf.keras.Input(shape=(chopsize, 4), name="quad_channel_input")
    
    # The Conv1D filters will naturally mix features across all 4 channels at every time step
    x = layers.Convolution1D(32, 4, padding='valid', kernel_regularizer=regularizers.l2(1e-4))(main_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.SpatialDropout1D(0.2)(x)
    
    x = layers.Convolution1D(64, 4, padding='valid', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Convolution1D(128, 4, padding='valid', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers to output the final Critic score
    dense = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    dense = layers.LayerNormalization()(dense)
    dense = layers.LeakyReLU(0.2)(dense)
    dense = layers.Dropout(0.3)(dense)
    
    output = layers.Dense(1)(dense)
    
    model = tf.keras.Model(inputs=main_input, outputs=output, name="Mixing_Discriminator")
    return model