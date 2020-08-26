from keras_applications.imagenet_utils import _obtain_input_shape
#from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils

import tensorflow as tf
keras = tf.keras
K = keras.backend



sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

WEIGHTS_PATH = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_NO_TOP = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Modular function for Fire Node

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    x = keras.layers.Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = keras.layers.Activation('relu', name=s_id + relu + sq1x1)(x)

    left = keras.layers.Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = keras.layers.Activation('relu', name=s_id + relu + exp1x1)(left)

    right = keras.layers.Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = keras.layers.Activation('relu', name=s_id + relu + exp3x3)(right)

    x = keras.layers.concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x


# Original SqueezeNet from paper.

def SqueezeNet(include_top=True, weights='imagenet',
               input_tensor=None, input_shape=None,
               pooling=None,
               classes=1000):
    """Instantiates the SqueezeNet architecture.
    """
        
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')


    input_shape = _obtain_input_shape(input_shape,
                                      default_size=227,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = keras.layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

#    graph = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='sigmoid')(input_layer)
#    graph = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(graph)
#    graph = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='sigmoid')(graph)
#    graph = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(graph)
#    graph = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='sigmoid')(graph)
#    graph = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(graph)
#    graph = keras.layers.Flatten()(graph)
#    graph = keras.layers.Dense(units=num_params, activation='tanh', bias_initializer='zeros', activity_regularizer='l2')(graph)


    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
    x = keras.layers.Activation('relu', name='relu_conv1')(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    
    if include_top:
        # It's not obvious where to cut the network... 
        # Could do the 8th or 9th layer... some work recommends cutting earlier layers.
        
        
        #x = fire_module(x, fire_id=6, squeeze=48, expand=192)
        #x = fire_module(x, fire_id=7, squeeze=48, expand=192)
        #x = fire_module(x, fire_id=8, squeeze=64, expand=256)
        #x = fire_module(x, fire_id=9, squeeze=64, expand=256)
        x = keras.layers.Dropout(0.5, name='drop9')(x)

        x = keras.layers.Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
        x = keras.layers.Activation('relu', name='relu_conv10')(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Activation('softmax', name='loss')(x)
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling=='max':
            x = keras.layers.GlobalMaxPooling2D()(x)
        elif pooling==None:
            pass
        else:
            raise ValueError("Unknown argument for 'pooling'=" + pooling)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = keras.Model(inputs, x, name='squeezenet')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
            
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


