"""
This network is build on top of SNGAN network implementation from: https://github.com/MingtaoGuo/sngan_projection_TensorFlow.git
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def upsampling(inputs):
    H = inputs.shape[1]
    W = inputs.shape[2]
    return tf.image.resize(inputs, [H * 2, W * 2])


def downsampling(inputs):
    return tf.nn.avg_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")


def relu(inputs, name=None):
    return tf.nn.relu(inputs, name=name)


def tanh(inputs, name=None):
    return tf.nn.tanh(inputs, name=name)


def sigmoid(inputs, name=None):
    return tf.nn.sigmoid(inputs, name=name)


def softmax(inputs, name=None):
    return tf.nn.softmax(inputs, name=name)


def _l2normalize(v, eps=1e-12):
    """l2 normize the input vector."""
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def global_sum_pooling(inputs):
    inputs = tf.reduce_sum(inputs, [1, 2], keepdims=False)
    return inputs


def spectral_normalization(name, weights, num_iters=1, update_collection=None,
                           with_sigma=False):
    """Performs Spectral Normalization on a weight tensor.
    Specifically it divides the weight tensor by its largest singular value. This
    is intended to stabilize GAN training, by making the discriminator satisfy a
    local 1-Lipschitz constraint.
    Based on [Spectral Normalization for Generative Adversarial Networks][sn-gan]
    [sn-gan] https://openreview.net/pdf?id=B1QRgziT-
    Args:
    weights: The weight tensor which requires spectral normalization
    num_iters: Number of SN iterations.
    update_collection: The update collection for assigning persisted variable u.
                       If None, the function will update u during the forward
                       pass. Else if the update_collection equals 'NO_OPS', the
                       function will not update the u during the forward. This
                       is useful for the discriminator, since it does not update
                       u in the second pass.
                       Else, it will put the assignment in a collection
                       defined by the user. Then the user need to run the
                       assignment explicitly.
    with_sigma: For debugging purpose. If True, the fuction returns
                the estimated singular value for the weight tensor.
    Returns:
    w_bar: The normalized weight tensor
    sigma: The estimated singular value for the weight tensor.
    """
    w_shape = weights.shape.as_list()
    w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
    u = tf.get_variable(name + 'u', [1, w_shape[-1]],
                        initializer=tf.truncated_normal_initializer(),
                        trainable=False)
    u_ = u
    for _ in range(num_iters):
        v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = _l2normalize(tf.matmul(v_, w_mat))

    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat /= sigma
    if update_collection is None:
        with tf.control_dependencies([u.assign(u_)]):
            w_bar = tf.reshape(w_mat, w_shape)
    else:
        w_bar = tf.reshape(w_mat, w_shape)
        if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(u_))
    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar
    

class SpectralNormalization(tf.keras.layers.Layer):
    """
    Standalone Spectral Normalization layer that can be applied to any weight tensor.
    This is useful when you want to apply spectral normalization to custom layers.
    """
    
    def __init__(self, num_iters=1, name='spectral_norm', **kwargs):
        """
        Initialize the Spectral Normalization layer.
        
        Args:
            num_iters: Number of power iterations
            name: Layer name
            **kwargs: Additional arguments
        """
        super().__init__(name=name, **kwargs)
        self.num_iters = num_iters
        self._u_dict = {}  # Store u vectors for different weight tensors
    
    def _l2_normalize(self, x, axis=None, epsilon=1e-12):
        """L2 normalize a tensor."""
        return tf.nn.l2_normalize(x, axis=axis, epsilon=epsilon)
    
    def spectral_normalize_weights(self, weights, weight_name='default', training=True):
        """
        Apply spectral normalization to a weight tensor.
        
        Args:
            weights: Weight tensor to normalize
            weight_name: Unique name for the weight tensor
            training: Whether to update singular vectors
            
        Returns:
            Tuple of (normalized_weights, singular_value)
        """
        w_shape = list(weights.shape)
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])
        
        # Get or create u vector for this weight
        if weight_name not in self._u_dict:
            self._u_dict[weight_name] = self.add_weight(
                name=f'u_{weight_name}',
                shape=(1, w_shape[-1]),
                initializer='truncated_normal',
                trainable=False,
                dtype=weights.dtype
            )
        
        u = self._u_dict[weight_name]
        u_ = u
        
        # Power iteration
        for _ in range(self.num_iters):
            v_ = self._l2_normalize(tf.matmul(u_, w_mat, transpose_b=True))
            u_ = self._l2_normalize(tf.matmul(v_, w_mat))
        
        # Calculate singular value
        sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
        
        # Update u if training
        if training:
            u.assign(u_)
        
        # Normalize weights
        w_normalized = w_mat / tf.maximum(sigma, 1e-12)
        w_bar = tf.reshape(w_normalized, w_shape)
        
        return w_bar, sigma
    
    def call(self, weights, weight_name='default', training=None, with_sigma=False):
        """
        Apply spectral normalization to input weights.
        
        Args:
            weights: Weight tensor to normalize
            weight_name: Unique identifier for the weights
            training: Training mode flag
            with_sigma: Whether to return the singular value
            
        Returns:
            Normalized weights, optionally with singular value
        """
        w_bar, sigma = self.spectral_normalize_weight(weights, weight_name, training)
        
        if with_sigma:
            return w_bar, sigma
        else:
            return w_bar


"""def conditional_batchnorm(x, scope_bn, y=None, nums_class=None):
    # Batch Normalization
    # Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    # with tf.variable_scope(scope_bn):
        if y is None:
            beta = tf.get_variable(name=scope_bn + 'beta', shape=[x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
        else:
            beta = tf.get_variable(name=scope_bn + 'beta', shape=[nums_class, x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[nums_class, x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
            beta, gamma = tf.nn.embedding_lookup(beta, y), tf.nn.embedding_lookup(gamma, y)
            beta = tf.reshape(beta, [-1, 1, 1, x.shape[-1]])
            gamma = tf.reshape(gamma, [-1, 1, 1, x.shape[-1]])
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments', keep_dims=True)
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(tf.less(tf.constant(2), tf.constant(5)), mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed
"""


def conditional_batchnorm(x, scope_bn, y=None, nums_class=None):
    # Batch Normalization
    # Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    # with tf.variable_scope(scope_bn):
    input_shape = tf.shape(x)
    channels = x.shape[-1]

    if y is None:
        # Standard batch normalization
        beta = tf.Variable(
            tf.zeros([channels]), 
            trainable=True, 
            name=f'{scope_bn}_beta'
        )
        gamma = tf.Variable(
            tf.ones([channels]), 
            trainable=True, 
            name=f'{scope_bn}_gamma'
        )
    else:
        # Conditional batch normalization
        beta = tf.Variable(
            tf.zeros([nums_class, channels]), 
            trainable=True, 
            name=f'{scope_bn}_beta'
        )
        gamma = tf.Variable(
            tf.ones([nums_class, channels]), 
            trainable=True, 
            name=f'{scope_bn}_gamma'
        )    
        
        if y.dtype == tf.float32:
             y = tf.cast(y, tf.int32)
        # Lookup parameters for current batch
        beta = tf.nn.embedding_lookup(beta, y)
        gamma = tf.nn.embedding_lookup(gamma, y)
        
        # Reshape for broadcasting
        beta = tf.reshape(beta, [-1, 1, 1, channels])
        gamma = tf.reshape(gamma, [-1, 1, 1, channels])
    
    # Calculate batch statistics
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keepdims=True)
    
    # Create exponential moving average variables
    ema_mean = tf.Variable(
        tf.zeros_like(batch_mean), 
        trainable=False, 
        name=f'{scope_bn}_ema_mean'
    )
    ema_var = tf.Variable(
        tf.ones_like(batch_var), 
        trainable=False, 
        name=f'{scope_bn}_ema_var'
    )

    decay = 0.5   # EMA decay factor
    
    def mean_var_with_update():
        ema_mean.assign(decay * ema_mean + (1 - decay) * batch_mean)
        ema_var.assign(decay * ema_var + (1 - decay) * batch_var)
        
        return tf.identity(batch_mean), tf.identity(batch_var)
    
    # Replicate the original TF1 logic: always update (condition always true)
    mean, var = mean_var_with_update()

    # TODO: possible issue: update of ema_mean and ema_var even during the inference phase 
    # Apply batch normalization
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    
    return normed


def conv(nums_out, k_size, strides, is_sn=False):

    # Create the conv layer
    conv_layer = tf.keras.layers.Conv2D(
        filters=nums_out,
        kernel_size=k_size,
        strides=strides,
        padding='SAME',
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
    )

    if is_sn:
        conv_layer = tf.keras.layers.SpectralNormalization(conv_layer)
    # outputs = conv_layer(inputs)
    return conv_layer


def dense(nums_out, is_sn=False):
    dense_layer = tf.keras.layers.Dense(
        units=nums_out,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
    )
    if is_sn:
        dense_layer = tf.keras.layers.SpectralNormalization(dense_layer)
    return dense_layer #dense_layer(inputs)


# def Inner_product(inputs, y, nums_class, update_collection=None):
#     W = inputs.shape[-1]
#     V = tf.get_variable("V", [nums_class, W], initializer=tf.glorot_uniform_initializer())
#     V = tf.transpose(V)
#     V = spectral_normalization("embed", V, update_collection=update_collection)
#     V = tf.transpose(V)
#     temp = tf.nn.embedding_lookup(V, y)
#     temp = tf.reduce_sum(temp * inputs, axis=1, keep_dims=True)
#     return temp

def Inner_product(inputs, y, nums_class):
    """
    Functional implementation that creates and applies the layer
    """
    W = inputs.shape[-1]
    
    with tf.name_scope('inner_product'):
        # Create the weight matrix
        V = tf.Variable(
            tf.keras.initializers.GlorotUniform()(shape=[nums_class, W]),
            name="V",
            trainable=True
        )
        # Apply spectral normalization if needed
        # V = tf.keras.layers.SpectralNormalization(V)
        # V = tf.transpose(V)
        if y.dtype == tf.float32:
            y = tf.cast(y, tf.int32)
        # Embedding lookup and inner product
        temp = tf.nn.embedding_lookup(V, y)
        temp = tf.reduce_sum(temp * inputs, axis=1, keepdims=True)
        
        return temp


# def G_Resblock(name, inputs, nums_out, y, nums_class, update_collection=None, is_sn=False):
#     # with tf.variable_scope(name):
#         temp = tf.identity(inputs)
#         inputs = conditional_batchnorm(inputs, "bn1", y, nums_class)
#         inputs = relu(inputs)
#         inputs = upsampling(inputs)
#         # print(name, ' upsample ', inputs)
#         inputs = conv("conv1", inputs, nums_out, 3, 1, update_collection, is_sn=is_sn)
#         inputs = conditional_batchnorm(inputs, "bn2", y, nums_class)
#         inputs = relu(inputs)
#         inputs = conv("conv2", inputs, nums_out, 3, 1, update_collection, is_sn=is_sn)
#         # Identity mapping
#         temp = upsampling(temp)  # upsampling before conv in G
#         temp = conv("identity", temp, nums_out, 1, 1, update_collection, is_sn=is_sn)
#         return inputs + temp


class GResBlock((tf.keras.layers.Layer)):
    def __init__(self, nums_out, nums_class, is_sn=False, name='g_resblock',  **kwargs):
        super(GResBlock, self).__init__(name=name, **kwargs)
        self.nums_out = nums_out
        self.nums_class = nums_class
        self.is_sn = is_sn

        # Build layers
        self._build_layers()
    
    def _build_layers(self):
        self.identity = layers.Identity()
        # self.cb1 = ConditionalBatch()
        self.relu1 = layers.ReLu()
        self.upsampling = layers.Upsampling2D(size=(2, 2), interpolation='bilinear')
        self.conv1 = layers.Conv2D(self.nums_out, 3, 1, padding='same', name='conv1')       
        self.conv2 = layers.Conv2D(self.nums_out, 3, 1, padding='same', name='conv2')
        
  
    

def G_Resblock_Encoder(name, inputs, nums_out, y, nums_class, update_collection=None, is_sn=False):
    # with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = conditional_batchnorm(inputs, "bn1", y, nums_class)
        inputs = relu(inputs)
        inputs = downsampling(inputs)
        # print(name, ' down-sample ', inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, update_collection, is_sn=is_sn)
        inputs = conditional_batchnorm(inputs, "bn2", y, nums_class)
        inputs = relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, update_collection, is_sn=is_sn)
        # Identity mapping
        temp = downsampling(temp)  # downsampling before conv in G
        temp = conv("identity", temp, nums_out, 1, 1, update_collection, is_sn=is_sn)
        return inputs + temp


def D_Resblock(name, inputs, nums_out, update_collection=None, is_down=True, is_sn=True):
    # with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = relu(inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, update_collection, is_sn=is_sn)
        inputs = relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, update_collection, is_sn=is_sn)
        if is_down:
            inputs = downsampling(inputs)  # downsampling after 2nd conv in D
            # Identity mapping
            temp = conv("identity", temp, nums_out, 1, 1, update_collection,
                        is_sn=is_sn)  # replacing identity mapping with 1x1 conv
            temp = downsampling(temp)
        # else:
        #     temp = conv("identity", temp, nums_out, 1, 1, update_collection, is_sn=True)
        return inputs + temp


def D_FirstResblock(name, inputs, nums_out, update_collection, is_down=True, is_sn=True):
    # with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, update_collection=update_collection, is_sn=is_sn)
        inputs = relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, update_collection=update_collection, is_sn=is_sn)
        if is_down:
            inputs = downsampling(inputs)
            # Identity mapping
            temp = downsampling(temp)
            temp = conv("identity", temp, nums_out, 1, 1, update_collection=update_collection, is_sn=is_sn)
        return inputs + temp


# CSVAE related blocks
def Decoder_Block(name, inputs, nums_out):
    # with tf.variable_scope(name):
        inputs = upsampling(inputs)
        # print(name, ' upsample ', inputs)
        inputs = conv("conv1", inputs, nums_out, 5, 1)
        inputs = conditional_batchnorm(inputs, "bn1")
        inputs = relu(inputs)
        return inputs


def Encoder_Block(name, inputs, nums_out):
    # with tf.variable_scope(name):
        inputs = downsampling(inputs)
        # print(name, ' down-sample ', inputs)
        inputs = conv("conv1", inputs, nums_out, 5, 1)
        inputs = conditional_batchnorm(inputs, "bn1")
        inputs = relu(inputs)
        return inputs


def safe_log(inp):
    EPS = 1e-10
    return tf.math.log(inp + EPS)


def KL(mu1, logvar1, mu2, logvar2):
    """
    Calculates the KL divergence between two Gaussians
    See appendix here for a generalized version of this formula: https://arxiv.org/pdf/1312.6114.pdf
    """
    std1 = tf.exp(0.5 * logvar1)
    std2 = tf.exp(0.5 * logvar2)
    return tf.reduce_sum(
        safe_log(std2) - safe_log(std1) + 0.5 * (tf.exp(logvar1) + (mu1 - mu2) ** 2) / tf.exp(logvar2) - 0.5,
        axis=-1)




def _l2normalize(v, eps=1e-12):
    return v / (tf.norm(v) + eps)

def sn(name, weights, num_iters=1, u_var=None, with_sigma=False):
    w_shape = weights.shape
    w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channels]

    if u_var is None:
        # Initialize `u` only once outside the training loop
        u_var = tf.Variable(
            tf.random.truncated_normal([1, w_shape[-1]]),
            trainable=False,
            name=name + "_u"
        )

    u_ = u_var
    for _ in range(num_iters):
        v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = _l2normalize(tf.matmul(v_, w_mat))

    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat_sn = w_mat / sigma
    w_bar = tf.reshape(w_mat_sn, w_shape)

    # Optional: update u_var outside if needed
    # u_var.assign(u_)  ← do this in training loop if required

    if with_sigma:
        return w_bar, sigma, u_
    else:
        return w_bar, u_

    


def compare_spectral_normalizations():
    """Compare custom vs TensorFlow spectral normalization implementations."""
    
    print("=" * 80)
    print("SPECTRAL NORMALIZATION COMPARISON")
    print("=" * 80)
    
    # Create input tensor
    input_tensor = tf.random.normal([64, 224, 224, 3])
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor dtype: {input_tensor.dtype}")
    print()
    
    # ========================================================================
    # 1. BASELINE: Regular Conv2D without spectral normalization
    # ========================================================================
    print("1. BASELINE: Regular Conv2D (no spectral normalization)")
    print("-" * 60)
    
    conv_layer_baseline = conv(nums_out=64, k_size=3, strides=1)
    output_baseline = conv_layer_baseline(input_tensor)
    weights_baseline, biases_baseline = conv_layer_baseline.get_weights()
    
    print(f"Output shape: {output_baseline.shape}")
    print(f"Conv weights shape: {weights_baseline.shape}")
    print(f"Conv biases shape: {biases_baseline.shape}")
    
    # Calculate spectral norm of baseline weights
    w_mat = np.reshape(weights_baseline, [-1, weights_baseline.shape[-1]])
    u, s, vh = np.linalg.svd(w_mat, full_matrices=False)
    baseline_spectral_norm = s[0]  # Largest singular value
    print(f"Baseline spectral norm: {baseline_spectral_norm:.6f}")
    print()
    
    # ========================================================================
    # 2. CUSTOM SPECTRAL NORMALIZATION
    # ========================================================================
    print("2. CUSTOM SPECTRAL NORMALIZATION")
    print("-" * 60)
    
    # Create conv layer and get its weights
    conv_layer_custom = conv(nums_out=64, k_size=3, strides=1)
    _ = conv_layer_custom(input_tensor)  # Build the layer
    original_weights, original_biases = conv_layer_custom.get_weights()
    
    # Apply custom spectral normalization
    # custom_sn = SpectralNormalization(num_iters=1)
    
    # Time the custom implementation
    normalized_weights_custom, sigma_custom = sn(name='custom_snconv',
        weights=original_weights
    )
    
    print(f"Original weights shape: {original_weights.shape}")
    print(f"Normalized weights shape: {normalized_weights_custom.shape}")
    print(f"Estimated spectral norm (sigma): {sigma_custom[0][0]:.6f}")
    
    # Verify normalization worked
    w_mat_custom = tf.reshape(normalized_weights_custom, [-1, normalized_weights_custom.shape[-1]])
    u_custom, s_custom, vh_custom = tf.linalg.svd(w_mat_custom, full_matrices=False)
    actual_spectral_norm_custom = s_custom[0][0]
    print(f"Actual spectral norm after normalization: {actual_spectral_norm_custom.numpy():.6f}")
    print()
    
    # ========================================================================
    # 3. TENSORFLOW'S BUILT-IN SPECTRAL NORMALIZATION
    # ========================================================================
    print("3. TENSORFLOW'S BUILT-IN SPECTRAL NORMALIZATION")
    print("-" * 60)
    
    # Create conv layer with TF's spectral normalization
    base_conv_tf = conv(nums_out=64, k_size=3, strides=1)
    conv_layer_tf = tf.keras.layers.SpectralNormalization(base_conv_tf)
    
    # Time the TF implementation
    output_tf = conv_layer_tf(input_tensor, training=True)
    
    print(f"Output shape: {output_tf.shape}")
    
    # Get the normalized weights from TF implementation
    # Note: TF's implementation stores the original weights and applies normalization on-the-fly
    tf_weights, tf_biases, _ = conv_layer_tf.get_weights()
    print(f"TF weights shape: {tf_weights.shape}")
    print(f"TF biases shape: {tf_biases.shape}")
    
    # Calculate spectral norm of TF's weights (these might be the original weights)
    w_mat_tf = tf.reshape(tf_weights, [-1, tf_weights.shape[-1]])
    u_tf, s_tf, vh_tf = tf.linalg.svd(w_mat_tf, full_matrices=False)
    tf_spectral_norm = s_tf[0][0]
    print(f"TF weights spectral norm: {tf_spectral_norm.numpy():.6f}")
    print()
    
    # ========================================================================
    # 4. COMPARISON AND ANALYSIS
    # ========================================================================
    print("4. COMPARISON AND ANALYSIS")
    print("-" * 60)
    
    print(f"Baseline spectral norm:        {baseline_spectral_norm:.6f}")
    print(f"Custom normalized spectral norm: {actual_spectral_norm_custom.numpy():.6f}")
    print(f"TF implementation spectral norm: {tf_spectral_norm.numpy():.6f}")
    print()
    
    # Check if normalization worked (spectral norm should be close to 1.0)
    custom_normalized = abs(actual_spectral_norm_custom.numpy() - 1.0) < 0.01
    print(f"Custom normalization successful: {custom_normalized} (spectral norm ≈ 1.0)")


if __name__ == "__main__":
    input = tf.random.normal([64, 224, 224, 3])
    conv_layer = conv(nums_out=64, k_size=3, strides=1)
    output = conv_layer(input)
    conv_weights, conv_biases = conv_layer.get_weights()
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Run the comparison
    compare_spectral_normalizations()
