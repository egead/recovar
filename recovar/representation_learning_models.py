import tensorflow as tf
from numpy import pi
from tensorflow import keras
from recovar.config import BATCH_SIZE
from recovar.layers import AddNoise, NormalizeStd
from recovar.utils import demean, l2_normalize, l2_distance, l4_normalize
from recovar.layers import (Downsample,
                                     Upsample,
                                     UpsampleNoactivation,
                                     ResIdentity,
                                     Padding,
                                     MaskedConv)

@tf.keras.utils.register_keras_serializable()
class AutoencoderBlock(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(self, name="autoencoder_block", *args, **kwargs):
        super(AutoencoderBlock, self).__init__(name=name, **kwargs)

    def get_config(self):
        config = super(AutoencoderBlock, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.inp = keras.layers.InputLayer(
            (self.N_TIMESTEPS, self.N_CHANNELS), batch_size=BATCH_SIZE
        )

        self.down1 = Downsample(8, 15, name="down_1")  # 3000 -> 1500
        self.down2 = Downsample(16, 13, name="down_2")  # 1500 -> 750
        self.pad1 = Padding([1, 1])  # 750 -> 752
        self.down3 = Downsample(32, 11, name="down_3")  # 752 -> 376
        self.down4 = Downsample(64, 9, name="down_4")  # 376 -> 188
        self.down5 = Downsample(64, 7, name="down_5")  # 188 -> 94

        self.resid1 = ResIdentity(64, 5, name="resid_1")
        self.resid2 = ResIdentity(64, 5, name="resid_2")
        self.resid3 = ResIdentity(64, 5, name="resid_3")
        self.resid4 = ResIdentity(64, 5, name="resid_4")
        self.resid5 = ResIdentity(64, 5, name="resid_5")

        self.up1 = Upsample(32, 7, name="up_1")  # 94 -> 188
        self.up2 = Upsample(16, 9, name="up_2")  # 188 -> 376
        self.up3 = Upsample(8, 11, name="up_3")  # 376 -> 752
        self.crop1 = tf.keras.layers.Cropping1D(cropping=(1, 1))  # 752 -> 750
        self.up4 = Upsample(4, 13, name="up_4")  # 750 -> 1500
        self.up5 = UpsampleNoactivation(3, 15, name="up_5")

    def _encoder(self, x, training):
        x0 = x
        x1 = self.down1(x0, training=training)
        x2 = self.down2(x1, training=training)
        x2p = self.pad1(x2)
        x3 = self.down3(x2p, training=training)
        x4 = self.down4(x3, training=training)
        x5 = self.down5(x4, training=training)

        x = self.resid1(x5, training=training)
        x = self.resid2(x, training=training)
        x = self.resid3(x, training=training)
        x = self.resid4(x, training=training)
        x = self.resid5(x, training=training)

        return x, (x1, x2, x3, x4, x5)

    def _decoder(self, x, training):
        x = self.up1(x, training=training)
        x = self.up2(x, training=training)
        x = self.up3(x, training=training)
        x = self.crop1(x)
        x = self.up4(x, training=training)
        x = self.up5(x, training=training)

        return x

    def call(self, inputs, training=False):
        x = self.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        f, compressions = self._encoder(x, training=training)
        y = self._decoder(f, training=training)

        return f, y, compressions
    
@tf.keras.utils.register_keras_serializable()
class PickARSingle(keras.Model):
    def __init__(self, name="pick_ar_single", log_var_limit=25, *args, **kwargs):
        super(PickARSingle, self).__init__(name=name, **kwargs)
        self.log_var_limit = log_var_limit

    def get_config(self):
        config = super(PickARSingle, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self._input_shape = input_shape
        self.num_input_channels = input_shape[-1]

        self.masked_conv1 = MaskedConv(activation="leaky_relu",
                                       num_of_filters=4*self.num_input_channels,
                                       filter_kernel_size=3)
        self.masked_conv2 = MaskedConv(num_of_filters=2*self.num_input_channels,
                                       filter_kernel_size=3)

        self.create_gaussian_log_var()
        
    def call(self, x, training=False):
        log_p_per_dim = self._estimate_gaussian_log_p(x)
        
        xp = tf.pad(x, paddings=[[0, 0], [1, 0], [0, 0]])
        xp = xp[:, :-1, :]
        f = self.masked_conv1(xp, training=training)
        y = self.masked_conv2(f, training=training)
        
        mu = y[:, :, 0:self.num_input_channels]
        log_var = self.log_var_limit * tf.nn.tanh(y[:, :, self.num_input_channels:] / self.log_var_limit)

        # 1 / ((2pi)^d/2 |det(sigma)|^1/2) exp(-0.5 * (x-mu)^T sigma^{-1} (x-mu))
        term1 = -0.5 * tf.reduce_mean(tf.square(x - mu) / tf.exp(log_var), axis=-1)
        term2 = -0.5 * tf.reduce_mean(log_var, axis=-1)
        term3 = -0.5 * tf.math.log(2.0 * pi)
        log_p_cond_per_dim = term1 + term2 + term3

        return log_p_per_dim, log_p_cond_per_dim
    
    def _estimate_gaussian_log_p(self, x):
        log_var = self.log_var_limit * tf.nn.tanh(self.log_var / self.log_var_limit)
        var = tf.exp(log_var)
        x_demeaned = x - tf.expand_dims(self.means, axis=0)

        #Changed to reduce over channels to get shape (batch, timesteps) (HAD SHAPE MISMATCH ERROR BEFORE)
        term1 = -0.5 * tf.reduce_mean(tf.square(x_demeaned) / var, axis=-1)
        term2 = -0.5 * tf.reduce_mean(log_var, axis=-1)
        term3 = -0.5 * tf.math.log(2.0 * pi)

        log_p_per_dim = term1 + term2 + term3
        return log_p_per_dim
    
    def create_gaussian_log_var(self):
        initial_value = tf.random.normal(shape=self._input_shape[1:])
        self.log_var = tf.Variable(initial_value=initial_value,
                                   trainable=True,
                                   dtype=tf.float32)
        self.means = tf.Variable(initial_value=initial_value,
                                 trainable=True,
                                 dtype=tf.float32)
    
@tf.keras.utils.register_keras_serializable()
class PickARMultiple(keras.Model):
    def __init__(self, num_reps=5, name="pick_ar_multiple", *args, **kwargs):
        super(PickARMultiple, self).__init__(name=name, **kwargs)
        self.num_reps = num_reps
        
    def get_config(self):
        config = super(PickARMultiple, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self._input_shape = input_shape
        self._ar_pickers = []
        
        for i in range(self.num_reps):
            ar_picker_single = PickARSingle(name=f"pick_ar_single_{i}")
            x = tf.zeros(self._input_shape[i])
            ar_picker_single(x, training=False)
            
            self._ar_pickers.append(ar_picker_single)
            
    def call(self, inputs, training=False):
        log_ps = []
        log_p_conds = []
        
        for i in range(self.num_reps):
            x = inputs[i]
            log_p, log_p_cond = self._ar_pickers[i](x, training=training)
            log_ps.append(log_p)
            log_p_conds.append(log_p_cond)
            
        return log_ps, log_p_conds
    
@tf.keras.utils.register_keras_serializable()
class RepresentationLearningSingleAutoencoder(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(self, name="representation_learning_autoencoder", input_noise_std=1e-6, *args, **kwargs):
        super(RepresentationLearningSingleAutoencoder, self).__init__(name=name, **kwargs)
        self.input_noise_std = input_noise_std

    def get_config(self):
        config = super(RepresentationLearningSingleAutoencoder, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.inp = keras.layers.InputLayer(
            (self.N_TIMESTEPS, self.N_CHANNELS), batch_size=BATCH_SIZE
        )

        self.normalize1 = NormalizeStd(name="normalize_1")
        self.add_noise = AddNoise(stddev=self.input_noise_std)
        self.normalize2 = NormalizeStd(name="normalize_2")

        self.autoencoder = AutoencoderBlock("autoencoder_block")
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, inputs, training=False):
        x = self.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        x = self.normalize1(x)
        x = self.add_noise(x)
        x = self.normalize2(x)

        f, y = self.autoencoder(x, training=training)
        f = self.bn(f, training=training)

        self.add_loss(tf.reduce_mean(l2_distance(x, y), axis=0))

        return f, y


@tf.keras.utils.register_keras_serializable()
class RepresentationLearningDenoisingSingleAutoencoder(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(
        self,
        name="representation_learning_denoising_autoencoder",
        input_noise_std=1e-6,
        denoising_noise_std=2e-1,
        *args,
        **kwargs
    ):
        super(RepresentationLearningDenoisingSingleAutoencoder, self).__init__(name=name, **kwargs)
        self.input_noise_std = input_noise_std
        self.denoising_noise_std = denoising_noise_std

    def get_config(self):
        config = super(RepresentationLearningDenoisingSingleAutoencoder, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.inp = keras.layers.InputLayer(
            (self.N_TIMESTEPS, self.N_CHANNELS), batch_size=BATCH_SIZE
        )

        self.normalize1 = NormalizeStd(name="normalize_1")
        self.add_noise_input = AddNoise(
            name="add_noise_input", stddev=self.input_noise_std
        )
        self.normalize2 = NormalizeStd(name="normalize_2")

        self.add_noise_denoising = AddNoise(
            name="add_noise_denoising", stddev=self.denoising_noise_std
        )
        self.normalize_denoising = NormalizeStd(name="normalize_denoising")

        self.autoencoder = AutoencoderBlock("autoencoder_block")
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, inputs, training=False):
        x = self.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        x = self.normalize1(x)
        x = self.add_noise_input(x)
        x = self.normalize2(x)

        if training:
            x_noised = self.normalize_denoising(self.add_noise_denoising(x))
        else:
            x_noised = x

        f, y = self.autoencoder(x_noised, training=training)
        f = self.bn(f, training=training)

        self.add_loss(tf.reduce_mean(l2_distance(x, y), axis=0))

        return f, y


@tf.keras.utils.register_keras_serializable()
class RepresentationLearningMultipleAutoencoder(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(
        self,
        name="representation_learning_autoencoder_ensemble",
        input_noise_std=1e-6,
        eps=1e-27,
        *args,
        **kwargs
    ):
        super(RepresentationLearningMultipleAutoencoder, self).__init__(name=name, **kwargs)
        self.input_noise_std = input_noise_std
        self.eps = eps

    def get_config(self):
        config = super(RepresentationLearningMultipleAutoencoder, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.inp = keras.layers.InputLayer(
            (self.N_TIMESTEPS, self.N_CHANNELS), batch_size=BATCH_SIZE
        )

        self.normalize1 = NormalizeStd(name="normalize_1")
        self.add_noise = AddNoise(stddev=self.input_noise_std)
        self.normalize2 = NormalizeStd(name="normalize_2")

        self.autoencoder1 = AutoencoderBlock("autoencoder_block1")
        self.autoencoder2 = AutoencoderBlock("autoencoder_block2")
        self.autoencoder3 = AutoencoderBlock("autoencoder_block3")
        self.autoencoder4 = AutoencoderBlock("autoencoder_block4")
        self.autoencoder5 = AutoencoderBlock("autoencoder_block5")

        self.ar1 = PickARMultiple(name="ar_multiple_1")
        self.ar2 = PickARMultiple(name="ar_multiple_2")
        self.ar3 = PickARMultiple(name="ar_multiple_3")
        self.ar4 = PickARMultiple(name="ar_multiple_4")
        self.ar5 = PickARMultiple(name="ar_multiple_5")
        
        self.linear1 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="linear_1",
            trainable=True,
        )

        self.linear2 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="linear_2",
            trainable=True,
        )

        self.linear3 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="linear_3",
            trainable=True,
        )

        self.linear4 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="linear_4",
            trainable=True,
        )

        self.linear5 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="linear_5",
            trainable=True,
        )

        self.bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn2 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn3 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn4 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn5 = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def estimate_surprise(self, 
                                       logp_comps,
                                       logp_cond_comps):
        t = tf.linspace(0., 1., self.N_TIMESTEPS)
        average_surprise = tf.zeros(shape=[self.N_TIMESTEPS])
        
        for log_p_comp, log_p_cond_comp in zip(logp_comps, logp_cond_comps):
            sh = tf.shape(log_p_comp)
            n = sh[-1]
            t_comp = tf.linspace(0., 1., n)
            deltat = t[1] - t[0]
            oversampler_mask = tf.where(tf.abs(t_comp[:, None] - t[None, :]) < deltat, 
                                        1., 0.)
            mi_single = log_p_cond_comp - log_p_comp
            mi_single_oversampled = mi_single @ oversampler_mask
            average_surprise = average_surprise + mi_single_oversampled
    
        return average_surprise
        
    def call(self, inputs, training=False):
        x = self.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        x = self.normalize1(x)
        x = self.add_noise(x)
        x = self.normalize2(x)

        f1, y1, comp1 = self.autoencoder1(x, training=training)
        f2, y2, comp2 = self.autoencoder2(x, training=training)
        f3, y3, comp3 = self.autoencoder3(x, training=training)
        f4, y4, comp4 = self.autoencoder4(x, training=training)
        f5, y5, comp5 = self.autoencoder5(x, training=training)

        logp_comp1, logp_cond_comp1 = self.ar1(tuple(tf.stop_gradient(c) for c in comp1))
        logp_comp2, logp_cond_comp2 = self.ar2(tuple(tf.stop_gradient(c)for c in comp2))
        logp_comp3, logp_cond_comp3 = self.ar3(tuple(tf.stop_gradient(c) for c in comp3))
        logp_comp4, logp_cond_comp4 = self.ar4(tuple(tf.stop_gradient(c) for c in comp4))
        logp_comp5, logp_cond_comp5 = self.ar5(tuple(tf.stop_gradient(c) for c in comp5))
        
        # logp_(n; mu, sigma}(x) = -log (det(sigma)) -(x_n - mu)^T sigma (x-mu)
        # If sigma is diagonal, 
        # logp_(n; mu, sigma}(x) = 
        # -\alpha \sum log var - \sum_c (x_{nc} - mu_c) var_c (x_{nc}-mu_c)
        
        f1p = tf.transpose(
            tf.matmul(self.linear1, tf.stop_gradient(f1), transpose_b=True),
            perm=[0, 2, 1],
        )
        f2p = tf.transpose(
            tf.matmul(self.linear2, tf.stop_gradient(f2), transpose_b=True),
            perm=[0, 2, 1],
        )
        f3p = tf.transpose(
            tf.matmul(self.linear3, tf.stop_gradient(f3), transpose_b=True),
            perm=[0, 2, 1],
        )
        f4p = tf.transpose(
            tf.matmul(self.linear4, tf.stop_gradient(f4), transpose_b=True),
            perm=[0, 2, 1],
        )
        f5p = tf.transpose(
            tf.matmul(self.linear5, tf.stop_gradient(f5), transpose_b=True),
            perm=[0, 2, 1],
        )

        f1p = self.bn1(f1p, training=training)
        f2p = self.bn2(f2p, training=training)
        f3p = self.bn3(f3p, training=training)
        f4p = self.bn4(f4p, training=training)
        f5p = self.bn5(f5p, training=training)

        reconstruction_loss = self._get_reconstruction_loss(x, y1, y2, y3, y4, y5)
        ensemble_distance_loss = self._get_ensemble_distance_loss(
            f1p, f2p, f3p, f4p, f5p
        )
        all_log_ps = logp_comp1 + logp_comp2 + logp_comp3 + logp_comp4 + logp_comp5 
        all_log_p_conds = logp_cond_comp1 + logp_cond_comp2 + logp_cond_comp3 + logp_cond_comp4 + logp_cond_comp5 
        picker_loss = self._get_picker_loss(all_log_ps, all_log_p_conds)
        
        self.add_loss(reconstruction_loss + ensemble_distance_loss + picker_loss)

        if training:
            return f1p, f2p, f3p, f4p, f5p, y1, y2, y3, y4, y5
        else:
            surprise = self.estimate_surprise(all_log_ps, all_log_p_conds)
            pickability_score = tf.reduce_max(surprise) / tf.reduce_mean(surprise)
            pick_index = tf.argmax(surprise)

            return f1p, f2p, f3p, f4p, f5p, y1, y2, y3, y4, y5, surprise, pick_index, pickability_score

    def _get_ensemble_distance_loss(self, f1p, f2p, f3p, f4p, f5p):
        ensemble_distance_loss = (
            tf.reduce_mean(self._l2_after_channel_normalizing(f1p, f2p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f1p, f3p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f2p, f3p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f1p, f4p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f2p, f4p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f3p, f4p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f1p, f5p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f2p, f5p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f3p, f5p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f4p, f5p), axis=0)
        ) / 10.0

        return ensemble_distance_loss

    def _get_reconstruction_loss(self, x, y1, y2, y3, y4, y5):
        reconstruction_loss = (
            tf.reduce_mean(l2_distance(x, y1), axis=0)
            + tf.reduce_mean(l2_distance(x, y2), axis=0)
            + tf.reduce_mean(l2_distance(x, y3), axis=0)
            + tf.reduce_mean(l2_distance(x, y4), axis=0)
            + tf.reduce_mean(l2_distance(x, y5), axis=0)
        ) / 5.0

        return reconstruction_loss

    def _get_picker_loss(self, log_ps, log_p_conds):
        h_means = []
        for log_p in log_ps:
            h = -tf.reduce_mean(log_p, axis=0)
            h_mean = tf.reduce_mean(h)
            h_means.append(h_mean)
        
        for log_p_cond in log_p_conds:
            h = -tf.reduce_mean(log_p_cond, axis=0)
            h_mean = tf.reduce_mean(h)
            h_means.append(h_mean)
            
        h_grand_mean = tf.reduce_mean(tf.convert_to_tensor(h_means))
        
        return h_grand_mean
    
    def _l2_after_channel_normalizing(self, x, y):
        x = demean(x, axis=1)
        y = demean(y, axis=1)

        x = l2_normalize(x, axis=1)
        y = l2_normalize(y, axis=1)

        distance = tf.sqrt(tf.reduce_mean(tf.square(x - y), axis=[1, 2]))

        return distance
    

@tf.keras.utils.register_keras_serializable()
class RepresentationLearningMultipleAutoencoderL4(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(
        self,
        name="representation_learning_autoencoderl4_ensemble",
        input_noise_std=1e-6,
        eps=1e-27,
        *args,
        **kwargs
    ):
        super(RepresentationLearningMultipleAutoencoderL4, self).__init__(name=name, **kwargs)
        self.input_noise_std = input_noise_std
        self.eps = eps

    def get_config(self):
        config = super(RepresentationLearningMultipleAutoencoderL4, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.inp = keras.layers.InputLayer(
            (self.N_TIMESTEPS, self.N_CHANNELS), batch_size=BATCH_SIZE
        )

        self.normalize1 = NormalizeStd(name="normalize_1")
        self.add_noise = AddNoise(stddev=self.input_noise_std)
        self.normalize2 = NormalizeStd(name="normalize_2")

        self.autoencoder1 = AutoencoderBlock("autoencoder_block1")
        self.autoencoder2 = AutoencoderBlock("autoencoder_block2")
        self.autoencoder3 = AutoencoderBlock("autoencoder_block3")
        self.autoencoder4 = AutoencoderBlock("autoencoder_block4")
        self.autoencoder5 = AutoencoderBlock("autoencoder_block5")

        self.linear1 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="linear_1",
            trainable=True,
        )

        self.linear2 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="linear_2",
            trainable=True,
        )

        self.linear3 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="linear_3",
            trainable=True,
        )

        self.linear4 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="linear_4",
            trainable=True,
        )

        self.linear5 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="linear_5",
            trainable=True,
        )

        self.bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn2 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn3 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn4 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn5 = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, inputs, training=False):
        x = self.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        x = self.normalize1(x)
        x = self.add_noise(x)
        x = self.normalize2(x)

        f1, y1 = self.autoencoder1(x, training=training)
        f2, y2 = self.autoencoder2(x, training=training)
        f3, y3 = self.autoencoder3(x, training=training)
        f4, y4 = self.autoencoder4(x, training=training)
        f5, y5 = self.autoencoder5(x, training=training)

        f1p = tf.transpose(
            tf.matmul(self.linear1, tf.stop_gradient(f1), transpose_b=True),
            perm=[0, 2, 1],
        )
        f2p = tf.transpose(
            tf.matmul(self.linear2, tf.stop_gradient(f2), transpose_b=True),
            perm=[0, 2, 1],
        )
        f3p = tf.transpose(
            tf.matmul(self.linear3, tf.stop_gradient(f3), transpose_b=True),
            perm=[0, 2, 1],
        )
        f4p = tf.transpose(
            tf.matmul(self.linear4, tf.stop_gradient(f4), transpose_b=True),
            perm=[0, 2, 1],
        )
        f5p = tf.transpose(
            tf.matmul(self.linear5, tf.stop_gradient(f5), transpose_b=True),
            perm=[0, 2, 1],
        )

        f1p = self.bn1(f1p, training=training)
        f2p = self.bn2(f2p, training=training)
        f3p = self.bn3(f3p, training=training)
        f4p = self.bn4(f4p, training=training)
        f5p = self.bn5(f5p, training=training)

        reconstruction_loss = self._get_reconstruction_loss(x, y1, y2, y3, y4, y5)
        ensemble_distance_loss = self._get_ensemble_distance_loss(
            f1p, f2p, f3p, f4p, f5p
        )

        self.add_loss(reconstruction_loss + ensemble_distance_loss)

        return f1p, f2p, f3p, f4p, f5p, y1, y2, y3, y4, y5

    def _get_ensemble_distance_loss(self, f1p, f2p, f3p, f4p, f5p):
        ensemble_distance_loss = (
            tf.reduce_mean(self._l4_after_channel_normalizing(f1p, f2p), axis=0)
            + tf.reduce_mean(self._l4_after_channel_normalizing(f1p, f3p), axis=0)
            + tf.reduce_mean(self._l4_after_channel_normalizing(f2p, f3p), axis=0)
            + tf.reduce_mean(self._l4_after_channel_normalizing(f1p, f4p), axis=0)
            + tf.reduce_mean(self._l4_after_channel_normalizing(f2p, f4p), axis=0)
            + tf.reduce_mean(self._l4_after_channel_normalizing(f3p, f4p), axis=0)
            + tf.reduce_mean(self._l4_after_channel_normalizing(f1p, f5p), axis=0)
            + tf.reduce_mean(self._l4_after_channel_normalizing(f2p, f5p), axis=0)
            + tf.reduce_mean(self._l4_after_channel_normalizing(f3p, f5p), axis=0)
            + tf.reduce_mean(self._l4_after_channel_normalizing(f4p, f5p), axis=0)
        ) / 10.0

        return ensemble_distance_loss

    def _get_reconstruction_loss(self, x, y1, y2, y3, y4, y5):
        reconstruction_loss = (
            tf.reduce_mean(l2_distance(x, y1), axis=0)
            + tf.reduce_mean(l2_distance(x, y2), axis=0)
            + tf.reduce_mean(l2_distance(x, y3), axis=0)
            + tf.reduce_mean(l2_distance(x, y4), axis=0)
            + tf.reduce_mean(l2_distance(x, y5), axis=0)
        ) / 5.0

        return reconstruction_loss

    def _l4_after_channel_normalizing(self, x, y):
        x = demean(x, axis=1)
        y = demean(y, axis=1)

        x = l4_normalize(x, axis=1)
        y = l4_normalize(y, axis=1)

        distance = tf.sqrt(tf.sqrt(tf.reduce_mean(tf.square(tf.square(x - y)), axis=[1, 2])))

        return distance
