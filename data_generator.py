import itertools
import scipy
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from functional import seq

class CustomDataGenerator(Sequence):
    def __init__(self, hdf5_file, brain_idx, batch_size = 16, view = "axial", mode = 'train', horizontal_flip = False, 
                 vertical_flip = False, rotation_range = 0, zoom_range = 0., shuffle = True):
        self.data_storage = hdf5_file.root.data
        self.truth_storage = hdf5_file.root.truth

        total_brains = self.data_storage.shape[0]
        self.brain_idx = self.get_brain_idx(brain_idx, mode, total_brains)
        self.batch_size = batch_size

        if view == 'axial':
            self.view_axes = (0, 1, 2, 3)
        elif view == 'sagittal':
            self.view_axes = (2, 1, 0, 3)
        elif view == 'coronal':
            self.view_axes = (1, 2, 0, 3)
        else:
            ValueError(f'unknown input view => {view}')

        self.mode            = mode
        self.horizontal_flip = horizontal_flip
        self.vertical_flip   = vertical_flip
        self.rotation_range  = rotation_range
        self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        self.shuffle         = shuffle
        self.data_shape      = tuple(np.array(self.data_storage.shape[1:])[np.array(self.view_axes)])
        self.indexes = [(i, j)
                        for i in self.brain_idx for j in range(self.data_shape[0])]

        print(f'Using {len(self.brain_idx)} out of {total_brains} brains')
        print(f'({len(self.brain_idx) * self.data_shape[0]} out of {total_brains * self.data_shape[0]} 2D slices)')
        print(f'the generated data shape in "{view}" view: {self.data_shape[1:]}')
        print('-----'*10)

    @staticmethod
    def get_brain_idx(brain_idx, mode, total_brains):
        if mode == 'validation':
            brain_idx = np.array([i for i in range(total_brains) if i not in brain_idx])
        elif mode == 'train':
            brain_idx = brain_idx
        else:
            ValueError(f'unknown mode => {mode}')
        return brain_idx
    
    
    def __len__(self):
        return int(np.floor(len(self.brain_idx) / self.batch_size))
    
    def __getitem__(self, index):
        
        idx = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
        X_batch , Y_batch = self.data_load_and_preprocess(idx)
        return X_batch, Y_batch
    
    def on_epoch_end(self):
        self.indexes  =  [(i, j) for i in self.brain_idx for j in range(self.data_shape[0])]
        
        if self.mode == 'train' and self.shuffle:
            np.random.shuffle(self.indexes)
    
    
    
    def data_load_and_preprocess(self, idx):
        
        slice_batch, label_batch = zip(*(seq(idx)
            .map(lambda index: (index[0], index[1])) 
              .map(lambda i: self.read_data(i[0], i[1])) 
                .map(lambda slice_and_label: (self.normalize_modalities(slice_and_label[0]), slice_and_label[1]))
                    .map(lambda slice_and_label: np.concatenate((slice_and_label[0], slice_and_label[1]), axis= -1))
                        .map(lambda slice_and_label: self.apply_transform(slice_and_label, self.get_random_transform()))
                            .map(lambda slice_and_label: (slice_and_label[..., :4], to_categorical(slice_and_label[..., 4], 4)))
                                .to_list()))
                            
        return np.array(slice_batch), np.array(label_batch)
    
    
    def read_data(self, brain_number, slice_number):
        slice_ = self.data_storage[brain_number].transpose(self.view_axes)[slice_number]
        label_ = self.truth_storage[brain_number].transpose(self.view_axes[:3])[slice_number]     
        label_ = np.expand_dims(label_, axis = -1)   
        
        return slice_, label_
    
    
    def normalize_slice(self, slice):
        b = np.percentile(slice, 99)
        t = np.percentile(slice, 1)
        slice = np.clip(slice, t, b)

        if np.std(slice) != 0:
            slice = (slice - np.mean(slice)) / np.std(slice)
        return slice
    
    def normalize_modalities(self, Slice):
        
        normalized_slices = np.zeros_like(Slice).astype(np.float32)
        for slice_ix in range(4):
            normalized_slices[..., slice_ix] = self.normalize_slice(Slice[..., slice_ix])

        return normalized_slices
    
    def flip_axis(self, x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

    
        
    def get_random_transform(self):

        if self.rotation_range:
            theta = np.random.uniform(-self.rotation_range,
                                      self.rotation_range)
        else:
            theta = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(
                self.zoom_range[0], self.zoom_range[1], 2)

        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self.vertical_flip

        return {'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical,
                                'theta': theta,
                                'zx': zx,
                                'zy': zy}
        
    
    def apply_affine_transform(self, x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                               row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0):

        transform_matrix = None

        if theta != 0:
            theta = np.deg2rad(theta)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            if transform_matrix is not None:
                transform_matrix = np.dot(transform_matrix, shift_matrix)
            else:
                transform_matrix = shift_matrix

        if shear != 0:
            shear = np.deg2rad(shear)
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            if transform_matrix is not None:
                transform_matrix = np.dot(transform_matrix, shear_matrix)
            else:
                transform_matrix = shear_matrix

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            if transform_matrix is not None:
                transform_matrix = np.dot(transform_matrix, zoom_matrix)
            else:
                transform_matrix = zoom_matrix

        if transform_matrix is not None:
            h, w = x.shape[row_axis], x.shape[col_axis]
            transform_matrix = self.transform_matrix_offset_center(
                transform_matrix, h, w)
            x = np.rollaxis(x, channel_axis, 0)
            final_affine_matrix = transform_matrix[:2, :2]
            final_offset = transform_matrix[:2, 2]

            channel_images = [scipy.ndimage.interpolation.afffine_transfrom(
                x_channel,
                final_affine_matrix,
                final_offset,
                order=1,
                mode=fill_mode,
                cval=cval) for x_channel in x]
            x = np.stack(channel_images, axis=0)
            x = np.rollaxis(x, 0, channel_axis + 1)
        return x
    
    def apply_transform(self, x, transform_parameters):

        x = self.apply_affine_transform(x, transform_parameters.get('theta', 0),
                                   transform_parameters.get('tx', 0),
                                   transform_parameters.get('ty', 0),
                                   transform_parameters.get('shear', 0),
                                   transform_parameters.get('zx', 1),
                                   transform_parameters.get('zy', 1),
                                   row_axis=0,
                                   col_axis=1,
                                   channel_axis=2)
        if transform_parameters.get('flip_horizontal', False):
            x = self.flip_axis(x, 1)
        if transform_parameters.get('flip_vertical', False):
            x = self.flip_axis(x, 0)
        return x

    
    def transform_matrix_offset_center(matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        return np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    
    
