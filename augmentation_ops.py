import tensorflow as tf
import numpy as np
import SimpleITK as sitk
from scipy.ndimage.interpolation import shift

def __np_op_identity(**kwargs):
    def np_function(np_in):
         np_out = np_in
         return np_out
    return np_function

def __np_op_translate(**kwargs):
    def np_function(np_in):
        # Use RandomState here because tf ops have to be thread-safe, changing global numpy state would defeat that
        rng = np.random.RandomState(kwargs.get('seed', np.random.randint(2 ** 32 - 1)))
        out = []
        for np_image in np_in:
            random_shift_stddev = kwargs.get('shift-stddev', 10.0)
            random_shift_mean = kwargs.get('shift-mean', 0.0)

            shift_vals = [rng.normal(random_shift_mean, random_shift_stddev), rng.normal(random_shift_mean, random_shift_stddev), 0]

            out.append(shift(np_image, shift_vals, mode='reflect', order=1))

        return np.asarray(out)

    return np_function


def __np_op_elastic_deformation(**kwargs):
    def np_function(np_in):

        #Use RandomState here because tf ops have to be thread-safe, changing global numpy state would defeat that
        rng = np.random.RandomState(kwargs.get('seed', np.random.randint(2 ** 32 - 1)))

        out = []
        for np_image in np_in:
            original_shape = np_image.shape
            new_image = np.zeros(original_shape)

            #print('full image in: ' + str(original_shape))

            for channel in range(original_shape[2]):
                image = sitk.GetImageFromArray(np.squeeze(np_image[:, :, channel]))
                #print('channel image in: ' + str(np.squeeze(np_image[:, :, channel]).shape))
                image.SetSpacing([1, 1])

                splineOrder = kwargs.get('splineOrder', 3)
                deformStrength = kwargs.get('deformStrength', 10)
                meshSize = kwargs.get('meshSize', [5, 5])

                bspline = sitk.BSplineTransformInitializer(image, meshSize, splineOrder)

                random_vector = rng.randn(len(bspline.GetParameters()))
                displacements = random_vector * deformStrength

                bspline.SetParameters(displacements)

                resample = sitk.ResampleImageFilter()
                resample.SetTransform(bspline)
                resample.SetReferenceImage(image)
                np_output = sitk.GetArrayFromImage(resample.Execute(image))
                #print('np shape out: ' + str(np_output.shape))
                new_image[:, :, channel] = np_output

            out.append(new_image)

        return np.asarray(out, dtype=np.float32)
    return np_function




#This is just a template for py_func stuff, needed for SimpleITK elastic deformation
def tf_op_identity(tf_tensor, **kwargs):
     #np_function = __np_op_identity(**kwargs)
     #out = tf.py_func(np_function, [tf_tensor], tf.float32)
     #out.set_shape(tf_tensor.get_shape())
     return tf_tensor

def tf_op_random_translation(tf_tensor, **kwargs):
    np_function = __np_op_translate(**kwargs)
    out = tf.py_func(np_function, [tf_tensor], tf.float32)
    out.set_shape(tf_tensor.get_shape())

    return out

def tf_op_elastic_deformation(tf_tensor, **kwargs):
    np_function = __np_op_elastic_deformation(**kwargs)
    out = tf.py_func(np_function, [tf_tensor], tf.float32)
    out.set_shape(tf_tensor.get_shape())
    return out


def tf_op_additive_noise(tf_tensor, **kwargs):
    out = tf_tensor + tf.random_normal(tf.shape(tf_tensor),
                                       mean=kwargs.get('mean', 0.0),
                                       stddev=kwargs.get('stddev', 0.2),
                                       seed=kwargs.get('seed', np.random.randint(2**32 - 1)))

    return out



def tf_op_multiplicative_noise(tf_tensor, **kwargs):

    out = tf_tensor * (np.ones_like(tf_tensor) + tf.random_normal(tf.shape(tf_tensor),
                                       mean=kwargs.get('mean', 0.0),
                                       stddev=kwargs.get('stddev', 0.2),
                                       seed=kwargs.get('seed', np.random.randint(2**32 - 1))))

    return out

def tf_op_intensity_scaling(tf_tensor, **kwargs):

    rng = np.random.RandomState(kwargs.get('seed', np.random.randint(2 ** 32 - 1)))
    out = tf_tensor * (1 + rng.normal(loc=kwargs.get('mean', 0.0),
                                       scale=kwargs.get('stddev', 0.2)))

    return out

def tf_op_intensity_shift(tf_tensor, **kwargs):
    rng = np.random.RandomState(kwargs.get('seed', np.random.randint(2 ** 32 - 1)))
    out = tf_tensor + rng.normal(loc=kwargs.get('mean', 0.0),
                                       scale=kwargs.get('stddev', 0.2))

    return out

def tf_op_mirror_left_right(tf_tensor, **kwargs):
    out = tf.map_fn(lambda img: tf.image.random_flip_left_right(img, seed=kwargs.get('seed', np.random.randint(2**32 - 1))), tf_tensor)

    return out