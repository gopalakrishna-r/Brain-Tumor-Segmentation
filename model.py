from tensorpack.callbacks import ModelSaver, InferenceRunner, ScheduledHyperParamSetter, ScalarStats, ClassificationError
from tensorpack.train import launch_train_with_config, SimpleTrainer
from tensorpack.train.config import TrainConfig
from tensorpack.dataflow import dataset, imgaug, BatchData, AugmentImageComponent
from tensorpack.utils import logger
from tensorpack.graph_builder.model_desc import ModelDesc
from tensorpack.tfutils.sessinit import SmartInit
from tensorpack.tfutils.argscope import argscope
from tensorpack.models import (AvgPooling, BatchNorm, Conv2D, GlobalAvgPooling, FullyConnected)
from tensorpack.tfutils.common import get_global_step_var
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from unicodedata import name

import tensorflow.keras.backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import (Activation,
                                     concatenate, 
                                     GaussianNoise, GlobalAveragePooling2D,
                                     MaxPool2D, Permute, PReLU, Reshape,
                                     UpSampling2D, add, multiply)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from loss import custom_loss

K.set_image_data_format("channels_last")
import tensorflow as tf


def unet_model(input_shape, modified_unet = True, learning_rate = 0.01, start_channel = 64, 
               number_of_levels = 3, inc_rate = 2, output_channels = 4, saved_model_dir = None):
    
    input_layer = Input(shape = input_shape, name = "input_layer")
    name = 'unet'
    with argscope([Conv2D], padding='same' ):
            if modified_unet:
                x = GaussianNoise(0.01, name='GaussianNoise')(input_layer)
                x = Conv2D(f'{name}_conv_1.0', (x),  filters=64, kernel_size=2)
                x = level_block_modified('modified_lvl_blk', x, start_channel, number_of_levels, inc_rate)
                x = BatchNorm(f'{name}_BatchNorm', (x),  axis=3)
                x = PReLU(shared_axes=[1, 2])(x)
            else:
                x = level_block('lvl_blk', x, input_layer, start_channel,number_of_levels, inc_rate)

    x = Conv2D(f'{name}_conv_2.0' ,(x), output_channels, 1)
    output_layer = Activation('softmax')(x)
        
    model = Model(input_layer, output_layer)
        
    if modified_unet:
        print("Using modified UNet")
    else:
        print("Using standard UNet")
    
    if saved_model_dir:
        model.load_weights(saved_model_dir)
        print(f"Loaded weights from {saved_model_dir}")
    
    sgd = SGD(lr = learning_rate, momentum=0.9, decay=0.0)
    model.compile(optimizer=sgd, loss=custom_loss, metrics=['accuracy'])
    
    return model


def se_block(name, x, ratio = 16):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = x.shape[channel_axis]
    se_shape = (1, 1, filters)
    with argscope([FullyConnected], kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.1)):
            se = GlobalAveragePooling2D()(x)
            se = Reshape(se_shape)(se)
            se = FullyConnected(f'{name}_FCN_1.0', (se), filters // ratio, activation=tf.nn.relu)
            se = FullyConnected(f'{name}_FCN_2.0', (se), filters, activation=tf.nn.sigmoid)
            
            if K.image_data_format() == 'channels_first':
                se = Permute((3, 1, 2))(se)
            x = multiply([x, se])
    return x

    

def level_block(name, x, dim, level, inc):
    
    if level > 0:
        m = conv_layers(f'{name}_{level}_conv_layer_m', x, dim)

        x = MaxPool2D(pool_size=(2, 2))(m)
        x = level_block(f'{name}_{level - 1}_res1.0', x, int(dim * inc), level - 1, inc)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D((x), filters=dim, kernel_siz=2, padding='same')

        m = concatenate([x, m])
        x = conv_layers(f'{name}_{level}_conv_layer_x', m, dim)
    else:
        x = conv_layers(f'{name}_{level}_conv_layer_x', x, dim)
    return x



def level_block_modified(name, x, dim, level, inc):
    with argscope([Conv2D], kernel_size=2,  padding='same'):
            if level > 0:
                m = res_block(f'{level}_res1.0', x, dim, encoder_path=True)
                x = Conv2D(f'{name}_conv1.0', m, filters=int(inc * dim), stride=2)
                x = level_block_modified(f'{level - 1}', x, int(dim * inc), level - 1, inc)

                x = UpSampling2D(size=(2, 2))(x)
                x = Conv2D(f'{name}_conv2.0', x, filters=dim)

                m = concatenate([x, m])
                m = se_block(f'{level}_seblock2.0', m, 8)
                x = res_block(f'{level}_res2.0', m, dim, encoder_path=False)
            else:
                x = res_block(f'{level}_res1.0', x, dim, encoder_path=True)
            return x   
      
def conv_layers(name, x, dim):
        with argscope([Conv2D], filters=dim, kernel_size=3,  padding='same'):
            with argscope([Activation], Activation='relu'):
                x = Conv2D(f'{name}_conv_1.0', x)
                x = Activation()(x)

                x = Conv2D(f'{name}_conv_2.0', x)
                x = Activation()(x)
                return x


def res_block(name, x, dim, encoder_path = True):
        with argscope([Conv2D], filters=dim, kernel_size=3, padding='same'):
                with argscope([BatchNorm], axis= 3):
                    m = BatchNorm(f'{name}_BatchNorm_1.0', (x))
                    m = PReLU(shared_axes=[1, 2])(m)
                    m = Conv2D(f'{name}_conv_1.0', (m))

                    m = BatchNorm(f'{name}_BatchNorm_2.0', (x))
                    m = PReLU(shared_axes=[1, 2])(m)
                    m = Conv2D(f'{name}_conv_2.0', (m))

                    if not encoder_path:
                        x = Conv2D(f'{name}_conv_encoder', x,  kernel_size=1)
                    x = add([x, m])
                    return x
    