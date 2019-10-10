import os, sys

sys.path.append(os.getcwd())

import time
import functools

import numpy as np
import tensorflow as tf
import sklearn.datasets
import csv

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.small_imagenet
import tflib.ops.layernorm
import tflib.plot
from scipy.misc import imsave
import scipy
import util
from load_folder_images import load_image_and_segmentation_from_idlist
from unet import unet_train
from metrics import dice_score
import augmentation_ops
import gan_model

import itertools
from tensorflow.python.client import timeline
import SimpleITK as sitk



CRITIC_ITERS = 5  # How many iterations to train the critic for
# N_GPUS = 1  # Number of GPUs
BATCH_SIZE = 16  # Batch size. Must be a multiple of N_GPUS
ITERS = 200000  # How many iterations to train for
LAMBDA = 10  # Gradient penalty lambda hyperparameter
NOISE_DIM = 128
N_GPUS = 1
DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]
MODE = 'wgan-gp'  # dcgan, wgan, wgan-gp, lsgan
DIM = 32  # Model dimensionality
SMALLEST_IMAGE_DIM = 4
KERNEL_SIZE = 5
IMAGE_WIDTH = 256
#IMAGE_HEIGHT = 128
IMAGE_HEIGHT = 256
#IMAGE_CHANNELS = 3
IMAGE_CHANNELS = 1
# This should match so that the images are in the range of [-1 1]
# Images are first shifted, then scaled
#NUMBER_OF_SEGMENTATION_CLASSES = 34
NUMBER_OF_SEGMENTATION_CLASSES = 2
#IMAGE_INTENSITY_SHIFT = [-128, -(NUMBER_OF_SEGMENTATION_CLASSES - 1) / 2.0]
#IMAGE_INTENSITY_SCALE = [128, (NUMBER_OF_SEGMENTATION_CLASSES - 1) / 2.0]
IMAGE_INTENSITY_SHIFT = [-128, 0]
IMAGE_INTENSITY_SCALE = [128, 1]
#IMAGE_INTENSITY_SHIFT = [-128, -0.5]
#IMAGE_INTENSITY_SCALE = [128, 0.5]

OUTPUT_DIM = IMAGE_WIDTH * IMAGE_HEIGHT * (1 + IMAGE_CHANNELS)  # Number of pixels in each image
#FOLDER_BASE = 'unet_cityscapes'
FOLDER_BASE = 'unet_train_lung_gan'
FOLD_SUFFIX = 'fold_3'
FOLDER_PATH_CONCAT = FOLDER_BASE + '_' + str(IMAGE_WIDTH) + 'x' + str(IMAGE_HEIGHT) + 'x' + str(IMAGE_CHANNELS) + \
                     '_noise' + str(NOISE_DIM) + '_batch' + str(BATCH_SIZE) + '_dim' + str(DIM) + '_' + FOLD_SUFFIX

FOLDER_PATH = FOLDER_PATH_CONCAT
lib.print_model_settings(locals().copy())
# CHECKPOINT_PATH = os.path.join(FOLDER_PATH_CONCAT, 'checkpoints')
CHECKPOINT_NAME = 'model.ckpt'
UNET_CHECKPOINT_NAME = 'unet.ckpt'
ITERATIONS_NAME = 'iters.npy'
CSV_LOG_NAME = 'out.csv'
ITERS_BETWEEN_OUTPUTS = 50

# Select which net to train
TRAIN_GAN = False
TRAIN_UNET = True

def augment_images(images, segmentation_images, augmentation_functions_image, augmentation_functions_segmentation,
                                                  augmentation_params):
    for aug_op_img, aug_op_seg, aug_params in zip(augmentation_functions_image, augmentation_functions_segmentation,
                                                  augmentation_params):
        seed = np.random.randint(2 ** 32 - 1)

        images = aug_op_img(images, seed=seed, **aug_params)

        segmentation_images = aug_op_seg(segmentation_images, seed=seed, **aug_params)

    return images, segmentation_images

def images_segmentations_from_paths(base_folder_path, idlist_img, idlist_seg, img_folder, seg_folder,
                                    resized_image_size, shift_params, rescale_params, image_channels,
                                    force_to_grayscale, num_preprocess_threads=16, min_queue_examples=2560, batch_size=BATCH_SIZE,
                                    shuffle=False):
    idlist_img_name = os.path.join(base_folder_path, idlist_img)
    idlist_seg_name = os.path.join(base_folder_path, idlist_seg)
    img_folder_path = os.path.join(base_folder_path, img_folder)
    seg_folder_path = os.path.join(base_folder_path, seg_folder)

    idlist_tensor_img = util.string_tensor_from_idlist_and_path(idlist_img_name, img_folder_path)
    idlist_tensor_seg = util.string_tensor_from_idlist_and_path(idlist_seg_name, seg_folder_path)

    images, segmentation_images = load_image_and_segmentation_from_idlist(idlist_tensor_img, idlist_tensor_seg,
                                                                          batch_size, num_preprocess_threads, min_queue_examples,
                                                                          resized_image_size=resized_image_size,
                                                                          shift_params=shift_params,
                                                                          rescale_params=rescale_params,
                                                                          shuffle=shuffle,
                                                                          force_to_grayscale=force_to_grayscale,
                                                                          image_channels=image_channels)

    
    return images, segmentation_images


def main_training_loop(base_path,
                       idlist_image_train,
                       idlist_seg_train,
                       idlist_image_val,
                       idlist_seg_val,
                       img_folder,
                       seg_folder,
                       train_unet_with_GAN = False,
                       number_of_gan_samples = BATCH_SIZE,
                       number_of_real_samples = BATCH_SIZE,
                       load_gan_model_path = None,
                       load_unet_model_path = None,
                       augmentation_functions_image=None,
                       augmentation_functions_segmentation=None,
                       augmentation_params=None,
                       threshold_val = None,
                       folder_path=FOLDER_PATH,
                       save_checkpoints=True,
                       only_test=False):
    # Check mutable default args

    if augmentation_params == None:
        augmentation_params = [{}]
        
    #Reset everything from before
    tf.reset_default_graph()
    graph = tf.Graph()
    lib.delete_all_params()
    lib.delete_param_aliases()
    iteration = 0
    lib.plot.set(iteration)
    CHECKPOINT_PATH = os.path.join(folder_path, 'checkpoints')
    with graph.as_default():
        with tf.Session(graph=graph) as session:

          
            if number_of_real_samples != 0:
                images, segmentation_images = images_segmentations_from_paths(base_path,
                                                                          idlist_image_train,
                                                                          idlist_seg_train,
                                                                          img_folder,
                                                                          seg_folder,
                                                                          batch_size=number_of_real_samples,
                                                                          resized_image_size=[IMAGE_HEIGHT, IMAGE_WIDTH],
                                                                          shift_params=IMAGE_INTENSITY_SHIFT,
                                                                          rescale_params=IMAGE_INTENSITY_SCALE,
                                                                          image_channels=3,
                                                                          force_to_grayscale=(IMAGE_CHANNELS == 1),
                                                                          shuffle=True)
            else:
                images, segmentation_images = images_segmentations_from_paths(base_path,
                                                                              idlist_image_train,
                                                                              idlist_seg_train,
                                                                              img_folder,
                                                                              seg_folder,
                                                                              batch_size=number_of_gan_samples,
                                                                              resized_image_size=[IMAGE_HEIGHT,
                                                                                                  IMAGE_WIDTH],
                                                                              shift_params=IMAGE_INTENSITY_SHIFT,
                                                                              rescale_params=IMAGE_INTENSITY_SCALE,
                                                                              image_channels=3,
                                                                              force_to_grayscale=(IMAGE_CHANNELS == 1),
                                                                              shuffle=True)

            if augmentation_functions_segmentation != None and augmentation_functions_image != None:
                images, segmentation_images =   augment_images(images, segmentation_images, augmentation_functions_image, augmentation_functions_segmentation,
                                                          augmentation_params)

            val_images, val_segmentation_images = images_segmentations_from_paths(base_path,
                                                                                  idlist_image_val,
                                                                                  idlist_seg_val,
                                                                                  img_folder,
                                                                                  seg_folder,
                                                                                  batch_size=1,
                                                                                  num_preprocess_threads=1,
                                                                                  min_queue_examples=1,
                                                                                  resized_image_size=[IMAGE_HEIGHT,
                                                                                                      IMAGE_WIDTH],
                                                                                  shift_params=IMAGE_INTENSITY_SHIFT,
                                                                                  rescale_params=IMAGE_INTENSITY_SCALE,
                                                                                  image_channels=3,
                                                                                  force_to_grayscale=(IMAGE_CHANNELS == 1),
                                                                                  shuffle=True)
            # Get number of entries in idlist_image_val so we know how many images to process
            with open(os.path.join(base_path, idlist_image_val)) as f:
                file_names = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            file_names = [folder_path + x.strip() for x in file_names]

            num_val_images = len(file_names)
           
            segmentation_max = tf.reduce_max(segmentation_images)
            segmentation_min = tf.reduce_min(segmentation_images)

            segmentations_normalized = tf.multiply(tf.add(tf.div(
                tf.subtract(
                    segmentation_images,
                    tf.reduce_min(segmentation_images)
                ),
                tf.subtract(
                    segmentation_max,
                    segmentation_min
                )
            ), -0.5), 2)



            gan_input_plus_segmentation = tf.concat([images, segmentations_normalized], 3)

            Generator, Discriminator = gan_model.GeneratorAndDiscriminator()

            is_training = None  # tf.Variable(False, trainable=False)
            stats_iter_bn = tf.Variable(0, trainable=False)

            session.run(tf.initialize_variables([stats_iter_bn]))

            if number_of_gan_samples != 0:
                gen_train_op, disc_train_op, gen_cost, disc_cost = gan_model.build(session, gan_input_plus_segmentation, Generator,
                                                                                   Discriminator, MODE, LAMBDA, number_of_gan_samples,
                                                                                   DEVICES,
                                                                                   KERNEL_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT,
                                                                                   IMAGE_CHANNELS, NOISE_DIM,
                                                                                   OUTPUT_DIM, SMALLEST_IMAGE_DIM, DIM,
                                                                                   is_training, stats_iter_bn)
            else:
                gen_train_op, disc_train_op, gen_cost, disc_cost = gan_model.build(session, gan_input_plus_segmentation,
                                                                                   Generator,
                                                                                   Discriminator, MODE, LAMBDA,
                                                                                   number_of_real_samples,
                                                                                   DEVICES,
                                                                                   KERNEL_SIZE, IMAGE_WIDTH,
                                                                                   IMAGE_HEIGHT,
                                                                                   IMAGE_CHANNELS, NOISE_DIM,
                                                                                   OUTPUT_DIM, SMALLEST_IMAGE_DIM, DIM,
                                                                                   is_training, stats_iter_bn)

            if train_unet_with_GAN == True:
                gen_data, gen_data_orig_shape = Generator(number_of_gan_samples, KERNEL_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT,
                                                          IMAGE_CHANNELS,
                                                          NOISE_DIM, OUTPUT_DIM, SMALLEST_IMAGE_DIM,
                                                          DIM, MODE, reuse=True,
                                                          stats_iter=stats_iter_bn, is_training=is_training)

                # Make sure that Generator data (NCHW) is transposed correctly before it's used as NHWC again
                gen_data = tf.reshape(gen_data, (number_of_gan_samples, IMAGE_CHANNELS + 1, IMAGE_HEIGHT, IMAGE_WIDTH))
                gen_data = util.NCHW_to_NHWC(gen_data)


                gen_img = tf.squeeze(gen_data[:, :, :, 0:IMAGE_CHANNELS])

                if len(gen_img.shape) == 3:
                    gen_img = tf.expand_dims(gen_img, -1)


                gen_seg = gen_data[:, :, :, IMAGE_CHANNELS]

                #unnormalize segmentation data for unet
                gen_seg = tf.add(tf.multiply(tf.multiply(tf.add(gen_seg, 1), 0.5),
                                      tf.subtract(
                                            segmentation_max,
                                            segmentation_min
                                        )
                                      ), segmentation_min)

                #Threshold the generated GAN data.

                if threshold_val != None:
                    zeros_like_gen_seg = tf.zeros_like(gen_seg)
                    gen_seg = zeros_like_gen_seg + tf.to_float(tf.greater_equal(gen_seg, threshold_val))


                if len(gen_seg.shape) == 3:
                    gen_seg = tf.expand_dims(gen_seg, -1)

                if augmentation_functions_segmentation != None and augmentation_functions_image != None:
                    gen_img, gen_seg =   augment_images(gen_img, gen_seg, augmentation_functions_image,
                                   augmentation_functions_segmentation,
                                   augmentation_params)

                if number_of_real_samples != 0:
                    images = tf.concat([images, gen_img], 0)
                    segmentation_images = tf.concat([segmentation_images, gen_seg], 0)
                else:
                    images = gen_img
                    segmentation_images = gen_seg

                gan_input_plus_segmentation = tf.concat([images, segmentation_images], 3)



            unet_train_op, unet_loss_train, pred_train, unet_out_nchw, probs_train = unet_train.unet_ops(images,
                                                                                                         segmentation_images,
                                                                                                         num_classes=NUMBER_OF_SEGMENTATION_CLASSES,
                                                                                                         is_training=True,
                                                                                                         reuse_vars=None)

            _, unet_loss_val, pred_val, _, _ = unet_train.unet_ops(val_images,
                                                                   val_segmentation_images,
                                                                   num_classes=NUMBER_OF_SEGMENTATION_CLASSES,
                                                                   is_training=False,
                                                                   reuse_vars=True)

            if only_test == True:
                test_images, test_segmentation_images = images_segmentations_from_paths(base_path,
                                                                                      idlist_image_val,
                                                                                      idlist_seg_val,
                                                                                      img_folder,
                                                                                      seg_folder,
                                                                                      batch_size=1,
                                                                                      num_preprocess_threads=1,
                                                                                      min_queue_examples=1,
                                                                                      resized_image_size=[IMAGE_HEIGHT,
                                                                                                          IMAGE_WIDTH],
                                                                                      shift_params=IMAGE_INTENSITY_SHIFT,
                                                                                      rescale_params=IMAGE_INTENSITY_SCALE,
                                                                                      image_channels=3,
                                                                                      force_to_grayscale=(
                                                                                      IMAGE_CHANNELS == 1),
                                                                                      shuffle=False)


            session.run(tf.initialize_all_variables())



            # Start training queue
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)




            # Train loop


            # only restore model stuff
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='g')
            var_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='d'))

            saver = tf.train.Saver(var_list=var_list)

            var_list_unet = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='unet')

            unet_saver = tf.train.Saver(var_list=var_list_unet)

            if load_gan_model_path != None:
                saver.restore(session, load_gan_model_path)
                iteration = 0
                print("GAN Model restored.")
            else:
                iteration = 0


            if load_unet_model_path != None:
                unet_saver.restore(session, load_unet_model_path)
                iteration = 0
                print("UNet Model restored.")
            else:
                iteration = 0

            imgs_in = session.run(gan_input_plus_segmentation)
            for k in range(BATCH_SIZE):
                imgs_folder = os.path.join(folder_path, 'in/iter_%d/') % 0
                if not os.path.exists(imgs_folder):
                    os.makedirs(imgs_folder)

                img_channel = imgs_in[k][:, :, 0:IMAGE_CHANNELS]
                img_seg = imgs_in[k][:, :, IMAGE_CHANNELS]

                imsave(os.path.join(imgs_folder, 'img_%d.png') % k,
                       img_channel.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS).squeeze())
                imsave(os.path.join(imgs_folder, 'seg_%d.png') % k,
                       img_seg.reshape(IMAGE_HEIGHT, IMAGE_WIDTH))

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            if only_test == True:
                #Get number of entries in idlist_image_val so we know how many images to process
                with open(os.path.join(base_path, idlist_image_val)) as f:
                    file_names = f.readlines()

                file_names = [os.path.basename(x.strip()) for x in file_names]

                num_images = len(file_names)

                _, unet_loss_test, pred_test, _, _ = unet_train.unet_ops(test_images,
                                                                         test_segmentation_images,
                                                                         num_classes=NUMBER_OF_SEGMENTATION_CLASSES,
                                                                       is_training=False,
                                                                       reuse_vars=True)

                dice_scores_test = []

                seg_test_0_1 = tf.multiply(tf.add(test_segmentation_images, 1.0), 0.5) * (NUMBER_OF_SEGMENTATION_CLASSES - 1)

                dice_unet_test = dice_score(tf.squeeze(pred_test), tf.squeeze(test_segmentation_images), num_classes=NUMBER_OF_SEGMENTATION_CLASSES, session=session)

                test_imgs_out = []
                test_segs_out = []
                test_preds_out = []
                with open(folder_path + '/test_out.txt', mode='wt') as myfile:

                    for img_index in range(num_images):
                        dice_score_test_value, test_pred, test_img, test_seg = session.run([dice_unet_test, pred_test, test_images, test_segmentation_images])

                        test_imgs_out.append(test_img)
                        test_segs_out.append(test_seg)
                        test_preds_out.append(test_pred)

                        dice_scores_test.append(dice_score_test_value)
                        out_string = 'Dice ' + str(img_index + 1) + ' / ' + str(num_images) + ': ' + str(dice_score_test_value)

                        print(out_string)
                        myfile.write(out_string + '\n')

                    print('Average Dice: ' + str(np.asarray(dice_scores_test).mean()) + ' (stddev: ' +
                          str(np.asarray(dice_scores_test).std()) + ')')

                    myfile.write('Average Dice: ' + str(np.asarray(dice_scores_test).mean()) + ' (stddev: ' +
                          str(np.asarray(dice_scores_test).std()) + ')' + '\n')

                counter = 0

                if not os.path.exists(os.path.join(folder_path, 'test_out')):
                    os.makedirs(os.path.join(folder_path, 'test_out'))
                for img, seg, pred, filename in zip(test_imgs_out, test_segs_out, test_preds_out, file_names):
                    img = ((img + 0) * (255.99)).astype('int32')
                    seg = seg.astype('int32')
                    pred = pred.astype('int32')

                    imsave(os.path.join(os.path.join(folder_path, 'test_out'), 'img_%d.png') % counter,
                           np.squeeze(img))

                    scipy.misc.toimage(np.squeeze(seg), cmin=0, cmax=255).save(os.path.join(os.path.join(folder_path, 'test_out'), 'seg_%d.png') % counter)
                    scipy.misc.toimage(np.squeeze(pred), cmin=0, cmax=255).save(
                        os.path.join(os.path.join(folder_path, 'test_out'), filename), format='png')

                    counter = counter + 1

                coord.request_stop()
                coord.join(threads)

                return _, _, _, _

            # The dice score expects labels to be [0 1], but the labels were brought to the range of [-1 1]
            # Therefore we need to bring it back to [0 1]
            seg_0_1 = tf.multiply(tf.add(segmentation_images, 1.0), 0.5) * (NUMBER_OF_SEGMENTATION_CLASSES - 1)
            seg_val_0_1 = tf.multiply(tf.add(val_segmentation_images, 1.0), 0.5) * (
            NUMBER_OF_SEGMENTATION_CLASSES - 1)

            dice_unet_train = dice_score(pred_train, tf.squeeze(segmentation_images),
                                             num_classes=NUMBER_OF_SEGMENTATION_CLASSES)
            dice_unet_val = dice_score(pred_val, tf.squeeze(val_segmentation_images),
                                           num_classes=NUMBER_OF_SEGMENTATION_CLASSES)

            test_pred_1 = session.run(pred_train)
            test_seg_1 = session.run(segmentation_images)
            test_seg_0_1 = session.run(seg_0_1)
            test_rounded = session.run(tf.round(seg_0_1))

            loss_vals = []
            last_loss_avg = 9999999

            unet_val_loss_value = float('nan')
            dice_value_val = 0
            dice_value_train = 0
            unet_train_loss_value = float('nan')
            while iteration < ITERS:

                start_time = time.time()

                if TRAIN_GAN == True:
                    # Train generator
                    if iteration > 0:
                        _ = session.run(gen_train_op)

                    # Train critic
                    if (MODE == 'dcgan') or (MODE == 'lsgan'):
                        disc_iters = 1
                    else:
                        disc_iters = CRITIC_ITERS
                    for i in xrange(disc_iters):
                        _disc_cost, _ = session.run([disc_cost, disc_train_op])

                    lib.plot.plot('train disc cost', _disc_cost)

                if TRAIN_UNET == True:
                    dice_value_train = session.run(dice_unet_train)
                    unet_train_loss_value, _ = session.run([unet_loss_train, unet_train_op])

                    if iteration % 20 == 0:


                        dice_scores_val = []
                        val_losses = []
                        for img_index in range(num_val_images):
                            dice_value_val = session.run(dice_unet_val)
                            val_loss = session.run(unet_loss_val)
                            dice_scores_val.append(dice_value_val)
                            val_losses.append(val_loss)

                        print('Average Dice: ' + str(np.asarray(dice_scores_val).mean()) + ' (stddev: ' +
                              str(np.asarray(dice_scores_val).std()) + ')')
                        dice_value_val = np.asarray(dice_scores_val).mean()

                        unet_val_loss_value = np.asarray(val_losses).mean()

                        print('Average validation loss: ' + str(unet_val_loss_value))

                lib.plot.plot('train unet dice', dice_value_train)
                lib.plot.plot('train unet cost', unet_train_loss_value)
                lib.plot.plot('val unet cost', unet_val_loss_value)
                lib.plot.plot('val unet dice', dice_value_val)


                if iteration % ITERS_BETWEEN_OUTPUTS == 1:
                    t = time.time()

                    # Save Checkpoints / Iteration count
                    if not os.path.exists(CHECKPOINT_PATH):
                        os.makedirs(CHECKPOINT_PATH)

                    np.save(os.path.join(CHECKPOINT_PATH, ITERATIONS_NAME), iteration)


                    if TRAIN_GAN == True and save_checkpoints == True:
                        # Generate Samples
                        gan_model.generate_image(iteration, session, Generator, KERNEL_SIZE, OUTPUT_DIM, SMALLEST_IMAGE_DIM,
                                                 DIM, BATCH_SIZE, NOISE_DIM, DEVICES, IMAGE_WIDTH, IMAGE_HEIGHT,
                                                 IMAGE_CHANNELS, MODE, folder_path, stats_iter_bn, is_training)
                        saver.save(session, os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME))

                    if TRAIN_UNET == True and save_checkpoints == True:
                        util.save_tensor_image_to_folder(session,
                                                         pred_train,
                                                         BATCH_SIZE,
                                                         iteration,
                                                         os.path.join(folder_path, 'pred_train'))
                        util.save_tensor_image_to_folder(session,
                                                         pred_val,
                                                         1,
                                                         iteration,
                                                         os.path.join(folder_path, 'pred_val'))
                        unet_saver.save(session, os.path.join(CHECKPOINT_PATH, UNET_CHECKPOINT_NAME))

                if iteration > 10000 and TRAIN_GAN == True:

                    loss_vals.append(_disc_cost)

                    if len(loss_vals) > 100:
                        avg_loss = np.asarray(loss_vals).mean()

                        if avg_loss > last_loss_avg:
                            print('Loss increased over the last 100 iterations, stopping!')
                            break
                        else:
                            last_loss_avg = avg_loss

                        del loss_vals[:]

                if iteration > 500 and TRAIN_UNET == True:
                    # After 3000 iters, check if the loss is decreasing still
                    loss_vals.append(unet_val_loss_value)

                    if len(loss_vals) > 50:
                        avg_loss = np.asarray(loss_vals).mean()

                        if avg_loss > last_loss_avg:
                            print('Loss increased over the last 50 iterations, stopping!')
                            lib.plot.flush(os.path.join(folder_path, CSV_LOG_NAME))
                            break
                        else:
                            last_loss_avg = avg_loss

                        del loss_vals[:]

                lib.plot.plot('time', time.time() - start_time)
                if (iteration < 5) or (iteration % 20 == 0):
                    #print('Before flush: dice: ' + str(dice_value_val) + ' loss: ' + str(unet_val_loss_value))
                    lib.plot.flush(os.path.join(folder_path, CSV_LOG_NAME))

                lib.plot.tick()
                iteration = iteration + 1

            coord.request_stop()
            coord.join(threads)

        return dice_value_train, dice_value_val, unet_val_loss_value, unet_train_loss_value


# Main grid search training
aug_ops_image = [augmentation_ops.tf_op_additive_noise,
                 augmentation_ops.tf_op_intensity_shift,
                 augmentation_ops.tf_op_intensity_scaling,
                 augmentation_ops.tf_op_elastic_deformation,
                 augmentation_ops.tf_op_random_translation]

aug_ops_segmentation = [augmentation_ops.tf_op_identity,
                        augmentation_ops.tf_op_identity,
                        augmentation_ops.tf_op_identity,
                        augmentation_ops.tf_op_elastic_deformation,
                        augmentation_ops.tf_op_random_translation]

log_out_total_validation_loss_and_dice = ['test', '1', '2', '3']

if not os.path.exists(FOLDER_PATH):
    os.makedirs(FOLDER_PATH)

additive_noise_vals = [0.00]
#intensity_shift_vals = [0.00, 0.05, 0.10]
#intensity_scaling_vals = [0.00, 0.05, 0.10]
#elastic_deformation_vals = [0.00, 5.00, 10.00]
#translation_vals = [0.00, 5.00, 10.00]

#intensity_shift_vals = [0.10]
#translation_vals = [5.00]
intensity_shift_vals = [0.00, 0.05]
intensity_scaling_vals = [0.00]
elastic_deformation_vals = [0.00]
translation_vals = [0.00, 5.00]
threshold_vals = [None]
numbers_of_samples = [ (0, BATCH_SIZE), (BATCH_SIZE / 2, BATCH_SIZE / 2), (BATCH_SIZE, 0) ]



total_length = len(numbers_of_samples) * len(threshold_vals) * len(additive_noise_vals) * len(intensity_scaling_vals) * len(intensity_shift_vals) * len(elastic_deformation_vals) * len(translation_vals)
counter = 1
time_format_string = time.strftime('%Y%m%d-%H%M%S')

TRAINING = True

if TRAINING:



    for (number_of_samples_real, number_of_samples_gan) in numbers_of_samples:
        aug_params = []
        folder_add = ""

        log_out_total_validation_loss_and_dice = []

        folder_add_mix = 'mix_' + str(number_of_samples_real) + "-" + str(number_of_samples_gan)

        if number_of_samples_gan == 0:
            train_with_gan = False
        else:
            train_with_gan = True

        for threshold_val in threshold_vals:


            folder_add_threshold =  '_thresh_' + str(threshold_val)

            for additive_noise_stdev in additive_noise_vals:#, 0.05, 0.1]:



                folder_add_noise = '_noise_std_' + str(additive_noise_stdev)
                aug_params.append({'mean': 0.0, 'stddev': additive_noise_stdev})


                for intensity_shift_stdev in intensity_shift_vals:
                    folder_add_shift = '_shift_std_' + str(intensity_shift_stdev)
                    aug_params.append({'mean': 0.0, 'stddev': intensity_shift_stdev})
                    for intensity_scaling_stdev in intensity_scaling_vals:
                        folder_add_scale = '_scaling_std_' + str(intensity_scaling_stdev)
                        aug_params.append({'mean': 0.0, 'stddev': intensity_scaling_stdev})
                        for elastic_deformation_strength in elastic_deformation_vals:
                            folder_add_elastic = '_deformStrength_' + str(elastic_deformation_strength)
                            aug_params.append(
                                {'splineOrder': 3, 'meshSize': [3, 3], 'deformStrength': elastic_deformation_strength})

                            for random_translation_strength in translation_vals:
                                folder_add_transliaton = '_translation_' + str(random_translation_strength)



                                aug_params.append(
                                    {'shift-stddev': elastic_deformation_strength})

                                folder_add = folder_add_mix + folder_add_threshold + folder_add_noise + folder_add_shift + folder_add_scale + folder_add_elastic \
                                             + folder_add_transliaton

                                if True:
                                    print('Training (' + str(counter) + '/' + str(total_length) + '): '+ folder_add)
                                    counter = counter + 1

                                    start = time.time()

                                    dice_value_train, dice_value_val, unet_val_loss_value, unet_train_loss_value = main_training_loop(
                                        base_path="/your_data_base_path/",
                                        idlist_image_train="train_f3.txt",
                                        idlist_seg_train="train_f3.txt",
                                        idlist_image_val="val_f3.txt",
                                        idlist_seg_val="val_f3.txt",
                                        img_folder="image/",
                                        seg_folder="segmentation/",
                                        augmentation_functions_image=aug_ops_image,
                                        augmentation_functions_segmentation=aug_ops_segmentation,
                                        augmentation_params=aug_params,
                                        folder_path= os.path.join(FOLDER_PATH, folder_add),
                                        load_gan_model_path='lung_gan_256x256x1_noise128_batch16_dim32_fold_3/noise_std_0.0_shift_std_0.0_scaling_std_0.0_deformStrength_0.0_translation_0.0/checkpoints/model.ckpt',
                                        save_checkpoints=True,
                                        train_unet_with_GAN=train_with_gan,
                                        number_of_gan_samples = number_of_samples_gan,# / 2,
                                        number_of_real_samples = number_of_samples_real,#BATCH_SIZE / 2,
                                        threshold_val=threshold_val,

                                        #number_of_real_samples = BATCH_SIZE,
                                        only_test=False
                                    )

                                    elapsed = time.time() - start

                                    print('Elapsed: ' + str(elapsed) + 's (ETA: ~' + str(elapsed * (total_length - counter)) + 's)')

                                    log_out_total_validation_loss_and_dice.append(folder_add + ': val_dice: ' +
                                                                                  str(dice_value_val) + ' val_loss: ' +
                                                                                  str(unet_val_loss_value))

                                    with open(FOLDER_PATH + '/out' + time_format_string + '.txt', mode='wt') as myfile:
                                        myfile.write('\n'.join(log_out_total_validation_loss_and_dice))

                                if True:
                                    test_prefix = 'TEST_'
                                    if not os.path.exists(test_prefix + FOLDER_PATH):
                                        os.makedirs(test_prefix + FOLDER_PATH)
                                    dice_value_train, dice_value_val, unet_val_loss_value, unet_train_loss_value = main_training_loop(
                                        base_path="/your_data_base_path/",
                                        idlist_image_train="train_f3.txt",
                                        idlist_seg_train="train_f3.txt",
                                        idlist_image_val="test_f3.txt",
                                        idlist_seg_val="test_f3.txt",
                                        img_folder="image/",
                                        seg_folder="segmentation/",
                                        folder_path=test_prefix + os.path.join(FOLDER_PATH, folder_add),
                                        save_checkpoints=False,
                                        only_test=True,
                                        #load_unet_model_path=os.path.join(FOLDER_PATH,
                                        #                                 'noise_std_0.0_shift_std_0.0_scaling_std_0.0_deformStrength_0.0_translation_0.0',
                                        #                                  'checkpoints', UNET_CHECKPOINT_NAME)
                                        load_unet_model_path=os.path.join(FOLDER_PATH, folder_add,'checkpoints', UNET_CHECKPOINT_NAME)
                                    )

                                    print('TEST Dice: ' + str(dice_value_val))

else:
    test_prefix = 'TEST_'
    if not os.path.exists(test_prefix + FOLDER_PATH):
        os.makedirs(test_prefix + FOLDER_PATH)
    dice_value_train, dice_value_val, unet_val_loss_value, unet_train_loss_value = main_training_loop(
        base_path="/your_data_base_path/",
        idlist_image_train="train_f1.txt",
        idlist_seg_train="train_f1.txt",
        idlist_image_val="test_f1.txt",
        idlist_seg_val="test_f1.txt",
        img_folder="image/",
        seg_folder="segmentation/",
        folder_path=test_prefix + os.path.join(FOLDER_PATH, 'unet-lung-fold1'),
        save_checkpoints=False,
        only_test=True,
        load_unet_model_path=os.path.join(FOLDER_PATH, 'noise_std_0.0_shift_std_0.0_scaling_std_0.0_deformStrength_0.0_translation_0.0','checkpoints', UNET_CHECKPOINT_NAME)
    )

    print('TEST Dice: ' + str(dice_value_val))