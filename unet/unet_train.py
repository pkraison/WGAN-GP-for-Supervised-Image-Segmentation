from net import unet_add
import tensorflow as tf
from util import NCHW_to_NHWC
from util import NHWC_to_NCHW

def fcn_loss(logits, labels, num_classes, head=None):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.upscore as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes
    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-4)
        labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

        softmax = tf.nn.softmax(logits) + epsilon

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax),
                                           head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(
                labels * tf.log(softmax), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')
        #tf.add_to_collection('losses', cross_entropy_mean)

        #loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        return cross_entropy_mean

def unet_ops(images, segmentation_images, lr=0.0001, depth=3, features=64, num_classes=2, is_training=True, reuse_vars=True):
    #with(tf.variable_scope('unet_main', reuse=reuse_vars)):
        images_nchw = NHWC_to_NCHW(images)

        unet_out = unet_add(images_nchw, depth, features, 'unet', num_classes=num_classes, use_batch_norm=False, is_training=is_training, reuse_vars=reuse_vars)

        label_list = []
        for class_val in range(num_classes):
            add_min = 0
            add_max = 0
            add_shift = -0.5
            if class_val == 0:
                add_min = -0.5

            if class_val == (num_classes - 1):
                add_max = 0.5

            lower_bound = add_shift + class_val
            upper_bound = add_shift + (class_val + 1)


            class_labels_tensor = tf.logical_and(
                    tf.greater_equal(segmentation_images, lower_bound),
                    tf.less(segmentation_images, upper_bound)
                )


            label_list.append(tf.to_float(class_labels_tensor))

        # Convert the boolean values into floats -- so that
        # computations in cross-entropy loss is correct

        # combined_mask = util_funcs.NCHW_to_NHWC(combined_mask)
        unet_out = NCHW_to_NHWC(unet_out)

        combined_mask = tf.concat(axis=3, values=label_list)

        # this uses inputs in NHWC format.
        loss = fcn_loss(unet_out, combined_mask, num_classes)


        probabilities = tf.nn.softmax(unet_out)

        #print('unet shape', unet_out)
        unet_out_nchw = NHWC_to_NCHW(unet_out)
        pred = tf.to_float(tf.argmax(unet_out_nchw, dimension=1))

        with tf.variable_scope("adam_vars_unet", reuse=reuse_vars):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            #optimizer = tf.train.MomentumOptimizer(learning_rate=1e-5, momentum=0.99, use_nesterov=True)

            # take care of possible batch norm updates
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                unet_train_op = optimizer.minimize(loss=loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='unet'))

        return unet_train_op, loss, pred, unet_out_nchw, probabilities
