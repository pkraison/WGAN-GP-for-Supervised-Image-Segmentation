import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.initializers import VarianceScaling

def conv2d(x, kernel_size, num_outputs, name, activation, weight_filler, stddev, use_batch_norm, is_training):
    x_shape = x.get_shape().as_list()
    num_inputs = x_shape[1]
    if weight_filler == 'he_normal':
        stddev = np.sqrt(2.0 / (kernel_size[0] * kernel_size[1] * num_outputs))

    initializer = VarianceScaling(scale=2.0, mode='fan_in')

    x = tf.layers.conv2d(x, num_outputs, kernel_size, data_format="channels_first", padding="SAME", kernel_initializer=initializer)

    if use_batch_norm:
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, data_format='NCHW', fused=True, scope=name+'/bn')

    if activation is not None:
        return activation(x)

    return x

def upsample2d(x, kernel_size, name=''):
    return tf.contrib.keras.layers.UpSampling2D(kernel_size, data_format='channels_first', name=name)(x)


def downsample2d(x, kernel_size, name=''):
    return tf.nn.avg_pool(x, ksize=[1, 1] + kernel_size, strides=[1, 1] + kernel_size, padding='SAME', data_format='NCHW', name=name)
