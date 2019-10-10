from PIL import Image
from skimage import color
from skimage import io
from scipy import misc
import imagehash
import tensorflow as tf
import numpy as np
import os
import scipy
from scipy.misc import imsave
import SimpleITK as sitk
import csv

def dhash(img_in, RESIZE_W=9, RESIZE_H=8):
    # Reduce size
    img = misc.imresize(img_in, (RESIZE_W, RESIZE_H))

    # Reduce color
    img = color.rgb2gray(img)

    # Compute the difference and Assign bits
    bits = []
    for row in xrange(img.shape[0]):
        for col in xrange(img.shape[1] - 1):
            l = img[row][col]
            r = img[row][col + 1]
            if l > r:
                bits.append('1')
            else:
                bits.append('0')

    dhash = ''
    for i in xrange(0, len(bits), 4):
        dhash += hex(int(''.join(bits[i:i + 4]), 2))

    return dhash


def string_tensor_from_idlist_and_path(idlist_path, folder_path, name=None):
    with open(idlist_path) as f:
        file_names = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    file_names = ([folder_path + x.strip() for x in file_names])

    string_tensor = tf.Variable(file_names, trainable=False,
                                name=name, validate_shape=False)
    return string_tensor


def unique_images(hash_dict, images):
    unique_image_list = []
    unique_image_seg = []

    for img in images:
        hash_val = imagehash.phash(Image.fromarray(img[:, :, 0]), hash_size=6)  # dhash(img)
        if hash_val not in hash_dict:
            unique_image_list.append(img[:, :, 0])
            unique_image_seg.append(img[:, :, 1])
            hash_dict[hash_val] = 1

    return unique_image_list, unique_image_seg


def average_image(path, size):
    # Access all PNG files in directory
    allfiles = os.listdir(path)
    imlist = [filename for filename in allfiles if filename[-4:] in [".png", ".PNG"]]

    # Assuming all images are the same size, get dimensions of first image

    w, h = Image.open(path + imlist[0]).size
    N = len(imlist)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr = np.zeros((h, w), np.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        imarr = np.array(Image.open(path + im), dtype=np.float)
        arr = arr + imarr / N

    # Round values in array and cast as 8-bit integer
    arr = np.array(np.round(arr), dtype=np.uint8)

    # Generate, save and preview final image
    out = Image.fromarray(arr)
    out = out.resize(size)
    out.save("Average.png")
    # out.show()
    return out


def mse_numpy_arrays(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension

    dif = imageA.astype("float") - imageB.astype("float")
    test = np.square(dif)
    err = (test).mean()  # mean_squared_error(imageA, imageB)
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def mse_average_list(avg, list):
    mse = 0;
    N = len(list)
    # avg_array = np.array(avg)
    for arr in list:
        new_mse = mse_numpy_arrays(np.squeeze(avg), np.squeeze(arr))
        # print(new_mse)
        mse = mse + new_mse / N
    return mse


def feature_vector_distance(feature_query, feature_test, dist_type="euc"):
    curr_dist = 0.0
    # Calculate Euclidean distance between two feature vectors
    if dist_type == "euc":
        curr_dist = scipy.spatial.distance.euclidean(feature_query, feature_test)
    # Calculate Cosine distance between two feature vectors
    if dist_type == "cos":
        curr_dist = scipy.spatial.distance.cosine(feature_query, feature_test)
    # Calculate Chevyshev distance between two feature vectors
    if dist_type == "chev":
        curr_dist = scipy.spatial.distance.chebyshev(feature_query, feature_test)

    return curr_dist


def write_nd_np(img, prefix, postfix):
    for i in range(img.shape[0]):
        write(sitk.GetImageFromArray(img[i, :, :, :]), prefix + '_c' + str(i) + postfix)


def write_np(img, path):
    write(sitk.GetImageFromArray(img), path)


def write(img, path):
    """
  Write a volume to a file path.

  :param img: the volume
  :param path: the target path
  :return:
  """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    writer = sitk.ImageFileWriter()
    writer.Execute(img, path, True)


def save_tensor_image_to_folder(session, tensor_to_save, batch_size, current_iter, folder_name, add_range=0.0,
                                mult_range=255.99):
    # _x_r = session.run(tensor_to_save)
    # _x_r = ((_x_r + 1.) * (255.99 / 2)).astype('int32')


    imgs = session.run(tensor_to_save)
    #  write_nd_np(imgs, os.path.join(folder_name, 'nd%d/') % current_iter, '.mha')
    imgs = ((imgs + add_range) * (mult_range)).astype('int32')

    for k in range(batch_size):
        imgs_folder = os.path.join(folder_name, '%d/') % current_iter
        if not os.path.exists(imgs_folder):
            os.makedirs(imgs_folder)

        if len(imgs.shape) == 4:
            img_channel = imgs[k][0][:, :]
            img_seg = imgs[k][1][:, :]
            imsave(os.path.join(imgs_folder, 'img_%d_0.png') % k,
                   np.squeeze(img_channel))
            imsave(os.path.join(imgs_folder, 'img_%d_1.png') % k,
                   np.squeeze(img_seg))

        else:
            img_channel = imgs[k][:, :]
            imsave(os.path.join(imgs_folder, 'img_%d.png') % k,
                   np.squeeze(img_channel))

        write_np(np.squeeze((img_channel / mult_range) - add_range), os.path.join(imgs_folder, 'img_%d.mha'))


def save_tensor_list_to_folder(session, tensor_list, batch_size, current_iter, folder_name):
    for t in tensor_list:
        save_tensor_image_to_folder(session, t, batch_size, current_iter, folder_name + t.name)


def write_output_csv(title_row, data_rows):
    with open("output.csv", "wb") as f:
        writer = csv.writer(f)
        rows = zip(*data_rows)
        writer.writerow(title_row)
        writer.writerows(rows)


def NHWC_to_NCHW(x):
    return tf.transpose(x, [0, 3, 1, 2])


def NCHW_to_NHWC(x):
    return tf.transpose(x, [0, 2, 3, 1])