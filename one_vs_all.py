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





POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}

print ("loading data .. ")

train_df = pickle.load(open("cache/train_df_256_aug.pik","rb"))
valid_df = pickle.load(open("cache/valid_df_256.pik","rb"))
silent_df = pickle.load(open("cache/silent_df_256.pik","rb"))
unknown_df = pickle.load(open("cache/unknown_df_256_aug.pik","rb"))
# test_df =  pickle.load(open("cache/test_df_256.pik","rb"))

silent_df = silent_df.sample(2100,random_state=1)
silent_df = pd.concat([silent_df]*5)


train_df.reset_index(inplace=True,drop=True)
valid_df.reset_index(inplace=True,drop=True)
unknown_df.reset_index(inplace=True,drop=True)
silent_df.reset_index(inplace=True,drop=True)

test_paths = glob(os.path.join('./data/', 'test/audio/*wav'))

unknown_df["one_class_label"] = 0
p = 0.


silent_df.label_id = name2id["silence"]
print(silent_df.shape)

full_train_df = pd.concat([train_df,silent_df,unknown_df])
full_train_df.reset_index(inplace=True,drop=True)








def train_generator(train_batch_size,selected_class ):
    


    while True:
        
        this_train = train_df.groupby('label_id').apply(lambda x: x.sample(n = 2000))
        extra_data_size = int(this_train.shape[0]* 0.1)

        this_train = pd.concat([this_train,
            silent_df.sample(extra_data_size) ,
                                unknown_df.sample(extra_data_size*2)],axis=0 )
        this_train["one_class_label"] = (this_train.label_id == selected_class).astype(int)
        

        this_train.reset_index(drop=True,inplace=True)
        shuffled_ids = random.sample(range(this_train.shape[0]), this_train.shape[0])
        
        for start in range(0, len(shuffled_ids), train_batch_size):
            x_batch = []
            y_batch = []
            
            end = min(start + train_batch_size, len(shuffled_ids))
            i_train_batch = shuffled_ids[start:end]
            for i in i_train_batch:
                x_batch.append(this_train.loc[i,'raw'].T)
                y_batch.append(this_train.one_class_label.values[i])
                
            x_batch = 1.- np.array(x_batch)/-80.
            y_batch = np.array(y_batch) #to_categorical(y_batch, num_classes = len(POSSIBLE_LABELS))
            
            yield x_batch, y_batch

def valid_generator(val_batch_size,selected_class):
    valid_df["one_class_label"] = (valid_df.label_id == selected_class).astype(int)

    while True:
        ids = list(range(valid_df.shape[0]))
        for start in range(0, len(ids), val_batch_size):
            x_batch = []
            y_batch = []
            end = min(start + val_batch_size, len(ids))
            i_val_batch = ids[start:end]
            for i in i_val_batch:
                x_batch.append(valid_df.loc[i,'raw'].T)
                y_batch.append(valid_df.one_class_label.values[i])

            x_batch = 1.- np.array(x_batch)/-80.
            y_batch = np.array(y_batch) #to_categorical(y_batch, num_classes = len(POSSIBLE_LABELS))
            yield x_batch, y_batch



def x_generator(test_df, test_batch_size,augment=False):
    while True:
        ids = list(range(test_df.shape[0]))
        
        for start in range(0, len(ids), test_batch_size):
            x_batch = []
            end = min(start + test_batch_size, len(ids))
            i_test_batch = ids[start:end]
#             this_paths = test_paths[start:end]
#             for x in this_paths:
            for i in i_test_batch:
                x_batch.append(test_df.loc[i,'raw'].T)
            
            x_batch = np.array(x_batch)
            x_batch = 1.- np.array(x_batch)/-80.
            yield x_batch



def batch_relu(x):
    x = BatchNormalization()(x)    
    x = Activation('relu')(x)
    
    return x 



timesteps, input_dim , latent_dim = 32,256, 128


def get_model():
 
    num_layers_perstack = 2 #np.random.randint(2, 5)
    first_kernel_size = 10 #np.random.randint(3, 6)
    first_num_filters = 64 #np.random.randint(64,84)
    num_dense = 128 #np.random.randint(128, 200)

    rate_drop_dense = 0.2 + np.random.rand() * 0.1
    optimizer_choice = "adam" # if np.random.random() < 0.5 else "sgd"
    # act = ['relu','elu']

    STAMP = 'freqconvs1d_%d_%d_%d_%d_%.2f'%(num_layers_perstack, first_kernel_size, first_num_filters, num_dense, \
            rate_drop_dense)


    ##### Model definition
    x_logml = Input(shape=(timesteps, input_dim)) #1 channel, 99 time, 161 freqs # S : np.ndarray [shape=(n_mels, t)]
    x_freq = Reshape((input_dim, timesteps))(x_logml)

    x = BatchNormalization()(x_freq)
    
    x = Conv1D(64,first_kernel_size,padding='same')(x)
    x = batch_relu(x)
    x = Conv1D(64,first_kernel_size,padding='same')(x)
    x = batch_relu(x)  
    
    x = Dropout(p/2)(x)
    x  = MaxPooling1D(2)(x)   


    x = Conv1D(128,3,padding='same')(x)
    x = batch_relu(x)
    x = Conv1D(128,3,padding='same')(x)
    x = batch_relu(x)

    x = Dropout(p/2)(x)    
    x  = MaxPooling1D(2)(x) 
    

    x = Conv1D(256,3,padding='same')(x)
    x = batch_relu(x)
    x = Conv1D(256,3,padding='same')(x)
    x = batch_relu(x)
    
    x = Dropout(p/2)(x)    
    

    x  = AveragePooling1D(2)(x) 
    x = GlobalMaxPooling1D()(x)



    x = Dense(1, activation = 'sigmoid', name='targets')(x)



    model = Model(inputs = x_logml, outputs = x)

    if optimizer_choice == "adam":
        optimizer = Adam(lr=1e-3)
    else:
        optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)




    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model, STAMP, optimizer_choice


if __name__=="__main__":
	# unknown_df = pickle.load(open("cache/unknown_df_256.pik","rb"))
    

    for selected_class in tqdm(range(11)):

        model , STAMP, optimizer_choice = get_model()

        batch_size = 64 #np.random.randint(64, 128)
        unknown_pct = 2 #np.random.randint(2,5)
        STAMP = "freqconv_2xunk_gmax_onevsall_class_{}".format(selected_class)#max_freqconvs_2510_avgshortcuts

        # STAMP = "freqconv_2xunk_gmax_class_{}".format(str(batch_size) ,str(unknown_pct))


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


        cw=  1 / (11 + 2.) 
        history = model.fit_generator(generator=train_generator(batch_size,selected_class),
                                      steps_per_epoch=train_df.shape[0]*(1./5)//batch_size,
                                      epochs=100,
                                      callbacks=callbacks,
                                      validation_data=valid_generator(batch_size,selected_class),
                                      validation_steps=int(np.ceil(valid_df.shape[0]/batch_size)) , 
                                      class_weight={0:cw, 1:(1-cw)})


        model.load_weights('./weights/{}.hdf5'.format(exp_name))
        bst_val_score = min(history.history['val_loss'])
        

        print ("Best val score: ", bst_val_score)



        print('Making training predictions ... ')
        train_predictions = model.predict_generator(x_generator(full_train_df, batch_size), 
                                                     int(np.ceil(len(full_train_df)/float(batch_size))), verbose=1)        
        np.save("cache/train_preds_{}.npy".format(exp_name),train_predictions)





        # print('Making test predictions ... ')
        # test_predictions = model.predict_generator(x_generator(test_df, batch_size), 
        #                                              int(np.ceil(len(test_df)/float(batch_size))), verbose=1)

        # np.save("cache/test_preds_{}.npy".format(exp_name),test_predictions)












