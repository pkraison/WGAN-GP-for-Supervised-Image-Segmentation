import tensorflow as tf
import util
import tflib as lib
from scipy.misc import imsave
import os
import numpy as np


def GeneratorAndDiscriminator():
    """
    Choose which generator and discriminator architecture to use by
    uncommenting one of these lines.
    """

    # Baseline (G: DCGAN, D: DCGAN)
    return DCGANGenerator, DCGANDiscriminator

    raise Exception('You must choose an architecture!')


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)


def Batchnorm(name, axes, inputs, MODE, is_training=True, stats_iter=None):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0, 2, 3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name, axes, inputs, fused=True, is_training=is_training,
                                           stats_iter=stats_iter)


def DCGANGenerator(n_samples, KERNEL_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, NOISE_DIM, OUTPUT_DIM,
                   SMALLEST_IMAGE_DIM,
                   dim, MODE, noise=None, bn=True, nonlinearity=tf.nn.relu, reuse=False, is_training=None,
                   stats_iter=None):
    with tf.variable_scope('g', reuse=reuse):

        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        if noise is None:
            noise = tf.random_normal([n_samples, NOISE_DIM])
        scale_factor = (IMAGE_WIDTH / IMAGE_HEIGHT)
        current_image_dim = SMALLEST_IMAGE_DIM
        current_num_filters = (IMAGE_WIDTH / (current_image_dim * scale_factor)) * dim
        output = lib.ops.linear.Linear('Generator.Input', noise.shape[1].value,
                                       scale_factor * current_image_dim * current_image_dim * current_num_filters,
                                       noise)
        output = tf.reshape(output, [-1, current_num_filters, current_image_dim, current_image_dim * scale_factor])

        if bn:
            output = Batchnorm('Generator.BN1', [0, 2, 3], output, MODE, is_training=is_training, stats_iter=stats_iter)
        output = nonlinearity(output)

        generator_stage = 2
        kernel_size = KERNEL_SIZE
        while (current_image_dim * scale_factor) < IMAGE_WIDTH / 2:

            output = lib.ops.deconv2d.Deconv2D('Generator.' + str(generator_stage), current_num_filters,
                                               current_num_filters / 2, kernel_size, output)

            if bn:
                output = Batchnorm('Generator.BN' + str(generator_stage), [0, 2, 3], output, MODE,
                                   is_training=is_training,
                                   stats_iter=stats_iter)
            output = nonlinearity(output)

            current_num_filters = current_num_filters / 2
            generator_stage = generator_stage + 1
            current_image_dim = current_image_dim * 2

        output = lib.ops.deconv2d.Deconv2D('Generator.' + str(generator_stage), current_num_filters, 1 + IMAGE_CHANNELS,
                                           kernel_size, output)

        output = tf.tanh(output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.reshape(output, [n_samples, OUTPUT_DIM]), output


def DCGANDiscriminator(inputs, KERNEL_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, NOISE_DIM, OUTPUT_DIM,
                       SMALLEST_IMAGE_DIM,
                       dim, MODE, bn=True, nonlinearity=LeakyReLU, reuse=False, is_training=None,
                       stats_iter=None):
    with tf.variable_scope('d', reuse=reuse):
        output = tf.reshape(inputs, [-1, 1 + IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH])

        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        scale_factor = (IMAGE_WIDTH / IMAGE_HEIGHT)
        current_image_dim = IMAGE_WIDTH
        current_num_filters = dim
        discriminator_stage = 1
        kernel_size = KERNEL_SIZE

        # output = lib.ops.conv2d.Conv2D('Discriminator.' + str(discriminator_stage), 1 + IMAGE_CHANNELS, current_num_filters, kernel_size, output, stride=2)

        output = lib.ops.conv2d.Conv2D('Discriminator.1', 1 + IMAGE_CHANNELS, current_num_filters, KERNEL_SIZE, output,
                                       stride=2)
        output = nonlinearity(output)

        current_image_dim = current_image_dim / 2
        current_num_filters = current_num_filters * 2
        discriminator_stage = discriminator_stage + 1

        while (current_image_dim > SMALLEST_IMAGE_DIM * scale_factor):

            output = lib.ops.conv2d.Conv2D('Discriminator.' + str(discriminator_stage), current_num_filters / 2,
                                           current_num_filters, kernel_size, output, stride=2)
            if bn:
                output = Batchnorm('Discriminator.BN' + str(discriminator_stage), [0, 2, 3], output,
                                   MODE, is_training=is_training, stats_iter=stats_iter)
            output = nonlinearity(output)

            current_image_dim = current_image_dim / 2
            current_num_filters = current_num_filters * 2
            discriminator_stage = discriminator_stage + 1

        output = tf.reshape(output, [-1, current_num_filters * SMALLEST_IMAGE_DIM])
        output = lib.ops.linear.Linear('Discriminator.Output', current_num_filters * SMALLEST_IMAGE_DIM, 1, output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.reshape(output, [-1])


def build(session, input_plus_segmentation, Generator, Discriminator, MODE, LAMBDA, BATCH_SIZE, DEVICES,
          KERNEL_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, NOISE_DIM,
          OUTPUT_DIM, SMALLEST_IMAGE_DIM, dim, is_training, stats_iter_bn):
    all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
    if tf.__version__.startswith('1.'):
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
    else:
        split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
    gen_costs, disc_costs, val_disc_costs = [], [], []
    for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
        with tf.device(device):

            real_data = tf.reshape(util.NHWC_to_NCHW(input_plus_segmentation),
                                   [BATCH_SIZE / len(DEVICES), OUTPUT_DIM])
            fake_data, _ = Generator(BATCH_SIZE / len(DEVICES), KERNEL_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS,
                                     NOISE_DIM, OUTPUT_DIM, SMALLEST_IMAGE_DIM,
                                     dim, MODE, stats_iter=stats_iter_bn, is_training=is_training)

            disc_real = Discriminator(real_data, KERNEL_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, NOISE_DIM,
                                      OUTPUT_DIM, SMALLEST_IMAGE_DIM,
                                      dim, MODE, stats_iter=stats_iter_bn, is_training=is_training)
            disc_fake = Discriminator(fake_data, KERNEL_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, NOISE_DIM,
                                      OUTPUT_DIM, SMALLEST_IMAGE_DIM,
                                      dim, MODE, stats_iter=stats_iter_bn, is_training=is_training)

            if MODE == 'wgan':
                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)


            elif MODE == 'wgan-gp':
                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
                alpha = tf.random_uniform(
                    shape=[BATCH_SIZE / len(DEVICES), 1],
                    minval=0.,
                    maxval=1.
                )
                differences = fake_data - real_data
                interpolates = real_data + (alpha * differences)
                gradients = tf.gradients(
                    Discriminator(interpolates, KERNEL_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, NOISE_DIM,
                                  OUTPUT_DIM, SMALLEST_IMAGE_DIM,
                                  dim, MODE, stats_iter=stats_iter_bn, is_training=is_training), [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                disc_cost += LAMBDA * gradient_penalty

            elif MODE == 'dcgan':
                try:  # tf pre-1.0 (bottom) vs 1.0 (top)
                    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                      labels=tf.ones_like(
                                                                                          disc_fake)))
                    disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                       labels=tf.zeros_like(
                                                                                           disc_fake)))
                    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                                        labels=tf.ones_like(
                                                                                            disc_real)))
                except Exception as e:
                    gen_cost = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
                    disc_cost = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
                    disc_cost += tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
                disc_cost /= 2.

            elif MODE == 'lsgan':
                gen_cost = tf.reduce_mean((disc_fake - 1) ** 2)
                disc_cost = (tf.reduce_mean((disc_real - 1) ** 2) + tf.reduce_mean((disc_fake - 0) ** 2)) / 2.

            else:
                raise Exception()

            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)

    gen_cost = tf.add_n(gen_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)

    if MODE == 'wgan':
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost,
                                                                              var_list=lib.params_with_name(
                                                                                  'Generator'),
                                                                              colocate_gradients_with_ops=True)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost,
                                                                               var_list=lib.params_with_name(
                                                                                   'Discriminator.'),
                                                                               colocate_gradients_with_ops=True)

        clip_ops = []
        for var in lib.params_with_name('Discriminator'):
            clip_bounds = [-.01, .01]
            clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
        clip_disc_weights = tf.group(*clip_ops)

    elif MODE == 'wgan-gp':
        gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                                                                                 var_list=lib.params_with_name(
                                                                                                     'Generator'),
                                                                                                 colocate_gradients_with_ops=True)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                                                                                  var_list=lib.params_with_name(
                                                                                                      'Discriminator.'),
                                                                                                  colocate_gradients_with_ops=True)

    elif MODE == 'dcgan':
        gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                                                                      var_list=lib.params_with_name(
                                                                                          'Generator'),
                                                                                      colocate_gradients_with_ops=True)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
                                                                                       var_list=lib.params_with_name(
                                                                                           'Discriminator.'),
                                                                                       colocate_gradients_with_ops=True)

    elif MODE == 'lsgan':
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(gen_cost,
                                                                              var_list=lib.params_with_name(
                                                                                  'Generator'),
                                                                              colocate_gradients_with_ops=True)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(disc_cost,
                                                                               var_list=lib.params_with_name(
                                                                                   'Discriminator.'),
                                                                               colocate_gradients_with_ops=True)

    else:
        raise Exception()

    return gen_train_op, disc_train_op, gen_cost, disc_cost


def generate_image(iteration, session, Generator, KERNEL_SIZE, OUTPUT_DIM, SMALLEST_IMAGE_DIM, dim, BATCH_SIZE,
                   NOISE_DIM, DEVICES, IMAGE_WIDTH, IMAGE_HEIGHT,
                   IMAGE_CHANNELS, MODE, folder_path, stats_iter_bn, is_training):
    # For generating samples
    fixed_noise = tf.random_normal(
        [BATCH_SIZE, NOISE_DIM])  # tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
    noise_input = tf.Variable(tf.zeros([BATCH_SIZE, NOISE_DIM]))
    n_samples = BATCH_SIZE / len(DEVICES)

    # NOTE: this session run is a hack as it doesn't care about devices and stuff
    noise_vec = session.run(fixed_noise)
    txt_folder = os.path.join(folder_path, 'out/iter_%d/') % iteration
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)
    np.savetxt(os.path.join(txt_folder, 'noise_input.txt'), noise_vec, fmt='%1.3f')

    session.run(noise_input.assign(noise_vec))

    gen_data, gen_data_orig_shape = Generator(n_samples, KERNEL_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS,
                                              NOISE_DIM, OUTPUT_DIM, SMALLEST_IMAGE_DIM,
                                              dim, MODE, reuse=True, noise=noise_input,
                                              stats_iter=stats_iter_bn, is_training=is_training)

    # Make sure that Generator data (NCHW) is transposed correctly before it's used as NHWC again
    gen_data = tf.reshape(gen_data, (BATCH_SIZE, IMAGE_CHANNELS + 1, IMAGE_HEIGHT, IMAGE_WIDTH))
    gen_data = util.NCHW_to_NHWC(gen_data)

    imgs_in = session.run(gen_data)

    imgs_in = ((imgs_in + 1.) * (255.99 / 2)).astype('int32')

    for k in range(BATCH_SIZE):
        imgs_folder = os.path.join(folder_path, 'out/iter_%d/') % iteration
        if not os.path.exists(imgs_folder):
            os.makedirs(imgs_folder)

        # segs_folder = os.path.join(folder_path, 'out/segs%d/') % iteration
        # if not os.path.exists(segs_folder):
        #    os.makedirs(segs_folder)

        img_channel = imgs_in[k][:, :, 0:IMAGE_CHANNELS]
        img_seg = imgs_in[k][:, :, IMAGE_CHANNELS]
        imsave(os.path.join(imgs_folder, 'img_%d.png') % k,
               img_channel.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS).squeeze())
        imsave(os.path.join(imgs_folder, 'seg_%d.png') % k,
               img_seg.reshape(IMAGE_HEIGHT, IMAGE_WIDTH))

    # interpolate between 2 random values / samples
    # interpolate between 2 values of output
    rand_val_1 = noise_vec[0, :].reshape(1, -1)
    rand_val_2 = noise_vec[1, :].reshape(1, -1)
    interpolation_steps = 10
    output_vals = []
    # first create noise batch, then generate images, then output
    noise_batch = noise_vec.copy()  # np.zeros([BATCH_SIZE, 128])
    for step in range(interpolation_steps + 1):
        noise_np_val = rand_val_2 * (step * 1.0 / interpolation_steps) + rand_val_1 * (
            1 - step * 1.0 / interpolation_steps)
        noise_batch[step, :] = noise_np_val
        output_vals.append(noise_np_val)

    session.run(noise_input.assign(noise_batch))

    # Run Generator again for interpolated images
    gen_data, _ = Generator(n_samples, KERNEL_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, NOISE_DIM, OUTPUT_DIM,
                            SMALLEST_IMAGE_DIM,
                            dim, MODE, reuse=True, noise=noise_input,
                            stats_iter=stats_iter_bn, is_training=is_training)

    # Again, account for NCHW stuff
    gen_data = tf.reshape(gen_data, (BATCH_SIZE, IMAGE_CHANNELS + 1, IMAGE_HEIGHT, IMAGE_WIDTH))
    gen_data = util.NCHW_to_NHWC(gen_data)
    imgs = session.run(gen_data)
    imgs = ((imgs + 1.) * (255.99 / 2)).astype('int32')

    for k in range(BATCH_SIZE):
        imgs_folder = os.path.join(folder_path, 'interp/iter_%d/') % iteration
        if not os.path.exists(imgs_folder):
            os.makedirs(imgs_folder)

        # segs_folder = os.path.join(folder_path, 'interp/segs%d/') % iteration
        # if not os.path.exists(segs_folder):
        #    os.makedirs(segs_folder)

        img_channel = imgs[k][:, :, 0:IMAGE_CHANNELS]
        img_seg = imgs[k][:, :, IMAGE_CHANNELS]
        imsave(os.path.join(imgs_folder, 'img_%d.png') % k,
               img_channel.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS).squeeze())
        imsave(os.path.join(imgs_folder, 'seg_%d.png') % k,
               img_seg.reshape(IMAGE_HEIGHT, IMAGE_WIDTH))

    np_output_array = np.squeeze(np.array(output_vals))
    np.savetxt(os.path.join(folder_path, 'interp/iter_%d/noise_input.txt') % iteration, np_output_array,
               fmt='%1.3f')
