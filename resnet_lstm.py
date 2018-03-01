from utils import *
import array 

from pydub import AudioSegment
import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Flatten, GlobalMaxPooling1D, MaxPooling2D, Activation, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPool2D, concatenate, Dense, Dropout
from keras.optimizers import Adam, SGD
from tensorflow.python.keras.utils import to_categorical

from keras.layers import Input, GRU, RepeatVector, BatchNormalization, TimeDistributed, Conv1D
from keras.layers import GlobalAveragePooling1D, LSTM, MaxPooling1D, CuDNNLSTM, Bidirectional
from keras import backend as K
from keras.layers import  Conv2D, MaxPooling2D, UpSampling2D, Lambda, Reshape
import keras
from keras.layers import AveragePooling1D, UpSampling1D

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, GRU, RepeatVector, BatchNormalization, TimeDistributed, Conv1D
from keras import backend as K
from keras.layers import  Conv2D, MaxPooling2D, UpSampling2D, Lambda, Reshape
from tqdm import tqdm
from keras.layers import ConvLSTM2D, Bidirectional, CuDNNLSTM, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D




POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}

print ("loading data .. ")

train_df = pickle.load( open("cache/train_df_256_aug.pik","rb"))
valid_df = pickle.load( open("cache/valid_df_256.pik","rb"))
silent_df = pickle.load(open("cache/silent_df_256.pik","rb"))
unknown_df = pickle.load(open("cache/unknown_df_256_aug.pik","rb"))
# test_df =  pickle.load(open("cache/test_df_256.pik","rb"))

train_df.reset_index(inplace=True)
valid_df.reset_index(inplace=True)
unknown_df.reset_index(inplace=True)
silent_df.reset_index(inplace=True)

# test_paths = glob(os.path.join('./data/', 'test/audio/*wav'))



def batch_relu(x):
    x = BatchNormalization()(x)    
    x = Activation('relu')(x)
    
    return x 



timesteps, input_dim , latent_dim = 32,256, 128



def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = keras.layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x






def train_generator(train_batch_size, unknown_portion):
    while True:
        
        this_train = train_df.groupby('label_id').apply(lambda x: x.sample(n = 2000))
        extra_data_size = int(this_train.shape[0]* 0.1)
        this_train = pd.concat([silent_df.sample(extra_data_size),
                                this_train,
                                unknown_df.sample(extra_data_size*unknown_portion)],axis=0 )
        
        this_train.reset_index(drop=True,inplace=True)
        
        shuffled_ids = random.sample(range(this_train.shape[0]), this_train.shape[0])
        
        for start in range(0, len(shuffled_ids), train_batch_size):
            x_batch = []
            y_batch = []
            
            end = min(start + train_batch_size, len(shuffled_ids))
            i_train_batch = shuffled_ids[start:end]
            for i in i_train_batch:
                x_batch.append(this_train.loc[i,'raw'].T)
#                 x_batch.append(process_wav_file(this_train.iloc[i], augment=True).T)

                y_batch.append(this_train.label_id.values[i])
                
            x_batch = 1.- np.array(x_batch)/-80.
            y_batch = to_categorical(y_batch, num_classes = len(POSSIBLE_LABELS))
            
            yield x_batch, y_batch


def valid_generator(val_batch_size):
    while True:
        ids = list(range(valid_df.shape[0]))
        for start in range(0, len(ids), val_batch_size):
            x_batch = []
            y_batch = []
            end = min(start + val_batch_size, len(ids))
            i_val_batch = ids[start:end]
            for i in i_val_batch:
                x_batch.append(valid_df.loc[i,'raw'].T)
                y_batch.append(valid_df.label_id.values[i])

            x_batch = 1.- np.array(x_batch)/-80.
            y_batch = to_categorical(y_batch, num_classes = len(POSSIBLE_LABELS))
            yield x_batch, y_batch


def test_generator(test_batch_size,augment=False):
    while True:
        ids = list(range(test_df.shape[0]))
        
        for start in range(0, len(ids), test_batch_size):
            x_batch = []
            end = min(start + test_batch_size, len(ids))
            i_test_batch = ids[start:end]
#             this_paths = test_paths[start:end]
#             for x in this_paths:
            for i in i_test_batch:
            #WATCHOUT > NO AUG
#                 x_batch.append(process_wav_file(x).T) #,reshape=False,augment=augment,pval=0.5))
                x_batch.append(test_df.loc[i,'raw'].T)

            x_batch = np.array(x_batch)
            x_batch = 1.- np.array(x_batch)/-80.
            
            yield x_batch


def ResNet50(img_input):
    

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(
        64, (7, 7), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    
    x = MaxPooling2D((1, 3))(x)

    x = conv_block(x, 3, [32, 32, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [32, 32, 64], stage=2, block='b')
    x = identity_block(x, 3, [32, 32, 64], stage=2, block='c')
    
    
    x = MaxPooling2D((1, 3))(x)

    
    

    x = conv_block(x, 3, [64, 64, 128], stage=3, block='a')
    x = identity_block(x, 3, [64, 64, 128], stage=3, block='b')
    x = identity_block(x, 3, [64, 64, 128], stage=3, block='c')
    x = identity_block(x, 3, [64, 64, 128], stage=3, block='d')

#     x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

#     x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((1, 2), name='avg_pool')(x)

#     x = GlobalMaxPooling2D()(x)
    x = Reshape((16,int(x.shape[-1]) * int(x.shape[-2])))(x)  #x = Reshape((16,9*128))(x)
    
    x = Bidirectional(CuDNNLSTM(128,return_sequences=False))(x)
    # Create model.
    return x



def get_model():
 
    num_layers_perstack = np.random.randint(2, 5)
    first_kernel_size = np.random.randint(7, 12)
    first_num_filters = np.random.randint(32,64)
    num_dense = np.random.randint(128, 250)

    rate_drop_dense = 0.1 + np.random.rand() * 0.1
    optimizer_choice = "adam" # if np.random.random() < 0.5 else "sgd"
    # act = ['relu','elu']

    STAMP = 'freqconvs1d_%d_%d_%d_%d_%.2f'%(num_layers_perstack, first_kernel_size, first_num_filters, num_dense, \
            rate_drop_dense)


    ##### Model definition
    x_logml = Input(shape=(timesteps, input_dim)) #1 channel, 99 time, 161 freqs # S : np.ndarray [shape=(n_mels, t)]

    x = Reshape((timesteps, input_dim, 1))(x_logml)
    # x = get_conv_stacks(x_logml)
    x = ResNet50(x)


    x = Dense(128, activation = 'relu')(x) #
    x = Dropout(0.3)(x)



    x = Dense(len(POSSIBLE_LABELS), activation = 'softmax', name='targets')(x)




    model = Model(inputs = x_logml, outputs = x)

    if optimizer_choice == "adam":
        optimizer = Adam(lr=1e-3)
    else:
        optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)




    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model, STAMP, optimizer_choice


if __name__=="__main__":
	# unknown_df = pickle.load(open("cache/unknown_df_256.pik","rb"))
    

    for i in tqdm(range(1)):

        model , STAMP, optimizer_choice = get_model()

        batch_size = 64 #np.random.randint(64, 128)
        unknown_pct = 2 # np.random.randint(2,11)

        STAMP = "conv2d_resnet_lstm" #{}_{}".format(str(batch_size) ,str(unknown_pct))


        exp_name = STAMP #max_freqconvs_2510_avgshortcuts
        callbacks = [EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   verbose=1),
                     ModelCheckpoint(monitor='val_loss',
                                     filepath='weights/{}.hdf5'.format(exp_name),
                                     save_best_only=True,
                                     save_weights_only=True)  ]


        if optimizer_choice == "adam":
            callbacks.append(ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       patience=3,
                                       verbose=1,
                                       epsilon=0.01,
                                      min_lr=1e-5))             


        print ("Beginning training for ",STAMP)

        history = model.fit_generator(generator=train_generator(batch_size,unknown_pct),
                                      steps_per_epoch=train_df.shape[0]*((1.1+0.1*unknown_pct)/5)//batch_size,
                                      epochs=100,
                                      callbacks=callbacks,
                                      validation_data=valid_generator(batch_size),
                                      validation_steps=int(np.ceil(valid_df.shape[0]/batch_size)))


        model.load_weights('./weights/{}.hdf5'.format(exp_name))
        bst_val_score = min(history.history['val_loss'])
        

        print ("Best val score: ", bst_val_score)

        # print('Making test predictions ... ')

        # predictions = model.predict_generator(test_generator(batch_size), int(np.ceil(len(test_paths)/float(batch_size))), verbose=1) #

        # np.save("cache/nn_massive_freq1d/predictions_{}-{}.npy".format(exp_name,bst_val_score),predictions)