import os, cv2, logging
import numpy as np
import keyNet.datasets.dataset_utils as tools
from tqdm import tqdm
from torch.utils.data import Dataset
from keyNet.aux.tools import check_directory

class pytorch_dataset(Dataset):
    def __init__(self, data, mode='train'):
        self.data =data

        ## Restrict the number of training and validation examples (9000 : 3000 = train : val)
        if mode == 'train':
            if len(self.data) > 9000:
                self.data = self.data[:9000]
        elif mode == 'val':
            if len(self.data) > 3000:
                self.data = self.data[:3000]

        logging.info('mode : {} the number of examples : {}'.format(mode, len(self.data)))        

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        im_src_patch, im_dst_patch, homography_src_2_dst, homography_dst_2_src = self.data[idx]
        # print(im_src_patch.shape, im_dst_patch.shape, homography_src_2_dst, homography_dst_2_src)

        return im_src_patch[0], im_dst_patch[0], homography_src_2_dst[0], homography_dst_2_src[0]


class DatasetGeneration(object):
    def __init__(self, dataset_root, savepair_root, size_patches, batch_size, max_angle, max_scaling, max_shearing, random_seed, is_debugging=False, load_tfrecord=True):

        self.size_patches = size_patches
        self.batch_size = batch_size
        self.dataset_root = dataset_root
        self.num_examples = 0
        self.num_val_examples = 0
        self.max_angle = max_angle
        self.max_scaling = max_scaling
        self.max_shearing = max_shearing
        self.is_debugging = is_debugging

        self.savepair_root = savepair_root

        ## Input lists
        self.training_data = [] ## input_image_pairs : self.input_image_pairs / self.src2dst_Hs / self.dst2src_Hs
        self.validation_data = []

        if load_tfrecord:
            self._load_tfrecord_images('keyNet/tfrecords/train_dataset.npz', is_val=False)
            self._load_tfrecord_images('keyNet/tfrecords/val_dataset.npz', is_val=True)

        else:
            if is_debugging:
                self.save_path = os.path.join(self.savepair_root ,'train_dataset_debug')
                self.save_val_path = os.path.join(self.savepair_root , 'val_dataset_debug')
            else:
                self.save_path = os.path.join(self.savepair_root , 'train_dataset')
                self.save_val_path = os.path.join(self.savepair_root , 'val_dataset')        

            savepair_exists = self.existence_check(self.savepair_root, is_debugging)

            if not savepair_exists:
                check_directory(self.save_path)
                check_directory(self.save_val_path)

                self.data_path = self._find_data_path(self.dataset_root)
                self.images_info = self._load_data_names(self.data_path)

                print("Total images in directory at \"" , self.dataset_root, "\" is : ", len(self.images_info))

                self._create_synthetic_pairs(is_val=False)
                self._create_synthetic_pairs(is_val=True)
            else:
                self._load_pair_images(is_val=False)
                self._load_pair_images(is_val=True)

        print("# of Training / validation : ", len(self.training_data), len(self.validation_data))

    def existence_check(self, root, is_debugging):
        if is_debugging:
            a = os.path.exists(os.path.join(self.savepair_root ,'train_dataset_debug'))
            b = os.path.exists(os.path.join(self.savepair_root , 'val_dataset_debug'))
        else:
            a = os.path.exists(os.path.join(self.savepair_root , 'train_dataset'))
            b = os.path.exists(os.path.join(self.savepair_root , 'val_dataset'))
        
        return a and b

    def get_training_data(self):
        return self.training_data

    def get_validation_data(self):
        return self.validation_data

    def _find_data_path(self, data_path):
        assert os.path.isdir(data_path), \
            "Invalid directory: {}".format(data_path)
        return data_path

    def _load_data_names(self, data_path):
        count = 0
        images_info = []

        for r, d, f in os.walk(data_path):
            for file_name in f:
                if file_name.endswith(".JPEG") or file_name.endswith(".jpg") or file_name.endswith(".png"):
                    images_info.append(os.path.join(data_path, r, file_name))
                    count += 1

        src_idx = np.random.permutation(len(np.asarray(images_info)))
        images_info = np.asarray(images_info)[src_idx]
        return images_info


    def _create_synthetic_pairs(self, is_val):

        self._create_pair_images(is_val)


    ## This is main dataset generation and preprocessing function.
    def _create_pair_images(self, is_val):
        # More stable repeatability when using bigger size patches
        if is_val:
            size_patches = 2 * self.size_patches
            self.counter += 1
        else:
            size_patches = self.size_patches
            self.counter = 0

        counter_patches = 0

        print('Generating Synthetic pairs . . .')
        
        save_path = self.save_val_path if is_val else self.save_path

        path_im_src_patch = os.path.join(save_path, 'im_src_patch')
        path_im_dst_patch = os.path.join(save_path, 'im_dst_patch')
        path_homography_src_2_dst = os.path.join(save_path, 'homography_src_2_dst')
        path_homography_dst_2_src = os.path.join(save_path, 'homography_dst_2_src')

        check_directory(path_im_src_patch)
        check_directory(path_im_dst_patch)
        check_directory(path_homography_src_2_dst)
        check_directory(path_homography_dst_2_src)

        for path_image_idx in tqdm(range(len(self.images_info))):
            name_image_path = self.images_info[(self.counter+path_image_idx) % len(self.images_info)]

            correct_patch = False
            counter = -1
            while counter < 10:

                counter += 1
                incorrect_h = True

                while incorrect_h:

                    scr_c = tools.read_color_image(name_image_path)

                    source_shape = scr_c.shape
                    h = tools.generate_composed_homography(self.max_angle, self.max_scaling, self.max_shearing)

                    inv_h = np.linalg.inv(h)
                    inv_h = inv_h / inv_h[2, 2]

                    scr = tools.to_black_and_white(scr_c)
                    dst = tools.color_distorsion(scr_c)
                    dst = tools.apply_h_2_source_image(dst, inv_h)

                    if dst.max() > 0.0:
                        incorrect_h = False

                scr_sobelx = cv2.Sobel(scr, cv2.CV_64F, 1, 0, ksize=3)
                scr_sobelx = abs(scr_sobelx.reshape((scr.shape[0], scr.shape[1], 1)))
                scr_sobelx = scr_sobelx.astype(float) / scr_sobelx.max()
                dst_sobelx = cv2.Sobel(dst, cv2.CV_64F, 1, 0, ksize=3)
                dst_sobelx = abs(dst_sobelx.reshape((dst.shape[0], dst.shape[1], 1)))
                dst_sobelx = dst_sobelx.astype(float) / dst_sobelx.max()

                scr = scr.astype(float) / scr.max()
                dst = dst.astype(float) / dst.max()

                if size_patches/2 >= scr.shape[0]-size_patches/2 or size_patches/2 >= scr.shape[1]-size_patches/2:
                    break

                window_point = [scr.shape[0]/2, scr.shape[1]/2]

                # Define points
                point_src = [window_point[0], window_point[1], 1.0]

                im_src_patch = scr[int(point_src[0] - size_patches / 2): int(point_src[0] + size_patches / 2),
                               int(point_src[1] - size_patches / 2): int(point_src[1] + size_patches / 2)]

                point_dst = inv_h.dot([point_src[1], point_src[0], 1.0])
                point_dst = [point_dst[1] / point_dst[2], point_dst[0] / point_dst[2]]

                if (point_dst[0] - size_patches / 2) < 0 or (point_dst[1] - size_patches / 2) < 0:
                    continue
                if (point_dst[0] + size_patches / 2) > source_shape[0] or (point_dst[1] + size_patches / 2) > \
                        source_shape[1]:
                    continue

                h_src_translation = np.asanyarray([[1., 0., -(int(point_src[1]) - size_patches / 2)],
                                                   [0., 1., -(int(point_src[0]) - size_patches / 2)], [0., 0., 1.]])
                h_dst_translation = np.asanyarray(
                    [[1., 0., int(point_dst[1] - size_patches / 2)], [0., 1., int(point_dst[0] - size_patches / 2)],
                     [0., 0., 1.]])

                im_dst_patch = dst[int(point_dst[0] - size_patches / 2): int(point_dst[0] + size_patches / 2),
                               int(point_dst[1] - size_patches / 2): int(point_dst[1] + size_patches / 2)]
                label_dst_patch = dst_sobelx[
                                  int(point_dst[0] - size_patches / 2): int(point_dst[0] + size_patches / 2),
                                  int(point_dst[1] - size_patches / 2): int(point_dst[1] + size_patches / 2)]
                label_scr_patch = scr_sobelx[
                                  int(point_src[0] - size_patches / 2): int(point_src[0] + size_patches / 2),
                                  int(point_src[1] - size_patches / 2): int(point_src[1] + size_patches / 2)]

                if im_src_patch.shape[0] != size_patches or im_src_patch.shape[1] != size_patches:
                    continue
                if label_dst_patch.max() < 0.25:
                    continue
                if label_scr_patch.max() < 0.25:
                    continue

                correct_patch = True
                break

            if correct_patch:
                im_src_patch = im_src_patch.reshape((1, im_src_patch.shape[0], im_src_patch.shape[1], 1))
                im_dst_patch = im_dst_patch.reshape((1, im_dst_patch.shape[0], im_dst_patch.shape[1], 1))

                homography = np.dot(h_src_translation, np.dot(h, h_dst_translation))

                homography_dst_2_src = homography.astype('float32')
                homography_dst_2_src = homography_dst_2_src.flatten()
                homography_dst_2_src = homography_dst_2_src / homography_dst_2_src[8]
                homography_dst_2_src = homography_dst_2_src[:8]

                homography_src_2_dst = np.linalg.inv(homography)
                homography_src_2_dst = homography_src_2_dst.astype('float32')
                homography_src_2_dst = homography_src_2_dst.flatten()
                homography_src_2_dst = homography_src_2_dst / homography_src_2_dst[8]
                homography_src_2_dst = homography_src_2_dst[:8]

                homography_src_2_dst = homography_src_2_dst.reshape((1, homography_src_2_dst.shape[0]))
                homography_dst_2_src = homography_dst_2_src.reshape((1, homography_dst_2_src.shape[0]))

                ## Save the patches by np format (For caching)
                name_image =  name_image_path.split('/')[-1]
                np.save(os.path.join(path_im_src_patch, name_image), im_src_patch)
                np.save(os.path.join(path_im_dst_patch, name_image), im_dst_patch)
                np.save(os.path.join(path_homography_src_2_dst, name_image), homography_src_2_dst)
                np.save(os.path.join(path_homography_dst_2_src, name_image), homography_dst_2_src)

                if is_val:
                    self.validation_data.append([im_src_patch, im_dst_patch, homography_src_2_dst, homography_dst_2_src])
                else:
                    self.training_data.append([im_src_patch, im_dst_patch, homography_src_2_dst, homography_dst_2_src])

                if self.is_debugging:
                    import matplotlib.pyplot as plt
                    print("Save : ", im_src_patch.shape, im_dst_patch.shape, homography_src_2_dst.shape, homography_dst_2_src.shape)
                    
                    """fig = plt.figure()
                    rows = 1 ; cols = 3
                    ax1 = fig.add_subplot(rows, cols, 1)
                    ax1.imshow(scr_c)
                    ax1.set_title('scr_c (input image)')
                    ax1.axis("off")

                    ax2 = fig.add_subplot(rows, cols, 2)
                    ax2.imshow(im_src_patch[0,:,:,0],  cmap='gray')
                    ax2.set_title('im_src_patch')
                    ax2.axis("off")

                    ax3 = fig.add_subplot(rows, cols, 3)
                    ax3.imshow(im_dst_patch[0,:,:,0],  cmap='gray')
                    ax3.set_title('im_dst_patch')
                    ax3.axis("off")

                    plt.show()"""

                counter_patches += 1

            ## Select the number of training patches and validation patches (and debug mode). (original paper : 9000, 3000)
            if is_val and counter_patches > 1500:
                break
            elif counter_patches > 4000:
                break
            if is_val and self.is_debugging and counter_patches > 100:
                break
            elif not is_val and self.is_debugging and counter_patches > 400:
                break

        self.counter = counter_patches

    def _load_pair_images(self, is_val):
        print('Loading Synthetic pairs . . .')
        
        save_path = self.save_val_path if is_val else self.save_path

        path_im_src_patch = os.path.join(save_path, 'im_src_patch')
        path_im_dst_patch = os.path.join(save_path, 'im_dst_patch')
        path_homography_src_2_dst = os.path.join(save_path, 'homography_src_2_dst')
        path_homography_dst_2_src = os.path.join(save_path, 'homography_dst_2_src')

        save_name_list= os.listdir(path_im_src_patch)

        for name_image in tqdm(save_name_list, total=len(save_name_list)):
            if name_image[-8:] != "JPEG.npy": 
                continue
            ## Load the patches by np format (For caching)
            im_src_patch = np.load(os.path.join(path_im_src_patch, name_image))
            im_dst_patch = np.load(os.path.join(path_im_dst_patch, name_image))
            homography_src_2_dst = np.load(os.path.join(path_homography_src_2_dst, name_image))
            homography_dst_2_src = np.load(os.path.join(path_homography_dst_2_src, name_image))

            if is_val:  
                self.validation_data.append([im_src_patch, im_dst_patch, homography_src_2_dst, homography_dst_2_src])
            else:
                self.training_data.append([im_src_patch, im_dst_patch, homography_src_2_dst, homography_dst_2_src])


    def _load_tfrecord_images(self, tfrecord_name, is_val=False):
        print('Loading Synthetic pairs . . .')
        load_dict = np.load(tfrecord_name)

        im_src_patch = load_dict['im_src_patch']
        im_dst_patch = load_dict['im_dst_patch']
        homography_src_2_dst = load_dict['homography_src_2_dst']
        homography_dst_2_src = load_dict['homography_dst_2_src']

        for a,b,c,d in zip(im_src_patch, im_dst_patch, homography_src_2_dst, homography_dst_2_src):
            if is_val:
                self.validation_data.append([a[np.newaxis, ...],b[np.newaxis, ...],c[np.newaxis, ...],d[np.newaxis, ...]])
            else:
                self.training_data.append([a[np.newaxis, ...],b[np.newaxis, ...],c[np.newaxis, ...],d[np.newaxis, ...]])

