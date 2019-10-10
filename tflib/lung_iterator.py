import numpy as np
import scipy.misc
import time
import os

def make_generator(idlist, img_folder, seg_folder, batch_size):
    epoch_count = [1]
    idlist_entries = []
    #load idlist, get all filenames
    with open(idlist) as f:
        idlist_entries = f.readlines()

    idlist_entries = [x.strip() for x in idlist_entries]


   # reader.SetFileName(os.path.join(self.image_directory, id_entry))

    def get_epoch():
        images = np.zeros((batch_size, 2, 128, 128), dtype='float')
        #files = range(n_files)
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(idlist_entries)
        epoch_count[0] += 1

        iter_count = batch_size

        num_files = len(idlist_entries)

        if num_files > batch_size:
            iter_count = num_files



        for i in range(iter_count):
            file = idlist_entries[i % num_files]
            path = os.path.join(img_folder, file)
            image = scipy.misc.imread(path).astype(np.float64)

            image = scipy.misc.imresize(image, [128, 128]).astype(np.float64)

            image = (image - 128.0) / 128.0

            seg_path = os.path.join(seg_folder, file)
            seg = scipy.misc.imread(seg_path).astype(np.float64)

            seg = scipy.misc.imresize(seg, [128, 128], interp="nearest").astype(np.float64)

            seg = (seg - 128.0) / 128.0

            images[i % batch_size, 0, :, :] = image
            images[i % batch_size, 1, :, :] = seg
            if i > 0 and i % (batch_size - 1) == 0:
                yield (images,)
    return get_epoch

def load(batch_size, data_dir_img='bla', data_dir_seg='bla', idlist_path='bla', val_dir_img='bla', val_dir_seg='bla', val_idlist_path='bla'):
    return (
        make_generator(idlist_path, data_dir_img, data_dir_seg, batch_size),
        make_generator(val_idlist_path, val_dir_img, val_dir_seg, batch_size)
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()