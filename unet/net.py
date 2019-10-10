import tensorflow as tf
from layers import (conv2d, upsample2d, downsample2d)
from tflib.ops.conv2d import Conv2D
from tensorflow.contrib.keras.api.keras.initializers import VarianceScaling

def unet_recursive(node, current_level, max_level, num_features, use_batch_norm, is_training):
    node_shape = node.get_shape().as_list()
    input_features = node_shape[1]
    
    initializer = VarianceScaling(scale=2.0, mode='fan_in')
    
    node = tf.layers.conv2d(node, num_features, 3, activation=tf.nn.relu, data_format='channels_first', padding='SAME', kernel_initializer=initializer)
    if current_level < max_level:

        node = tf.layers.conv2d(node, num_features, 3, activation=tf.nn.relu, data_format='channels_first', padding='SAME', kernel_initializer=initializer)
        downsample = downsample2d(node, [2, 2], name='downsample' + str(current_level))

        # recursion
        deeper_level = unet_recursive(downsample, current_level + 1, max_level, num_features, use_batch_norm, is_training)

        upsample = upsample2d(deeper_level, [2, 2], name='upsample' + str(current_level))

        node = tf.concat([node, upsample], axis=1, name='concat' + str(current_level))

        node = tf.layers.conv2d(node, num_features, 3, activation=tf.nn.relu, data_format='channels_first', padding='SAME', kernel_initializer=initializer)
        
    node = tf.layers.conv2d(node, num_features, 3, activation=tf.nn.relu, data_format='channels_first', padding='SAME', kernel_initializer=initializer)
    return node


def unet_add(node, levels, features_root, variable_scope, num_classes=2, use_batch_norm=False, is_training=False, reuse_vars=False):
    with tf.variable_scope(variable_scope, reuse=reuse_vars):
        all_outs = []
        node = unet_recursive(node, 0, levels, features_root, use_batch_norm, is_training)
        
        initializer = VarianceScaling(scale=2.0, mode='fan_in')
        node = tf.layers.conv2d(node, num_classes, 3, data_format='channels_first', padding='SAME', kernel_initializer=initializer)
        return node
