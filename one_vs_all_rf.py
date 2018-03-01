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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier



POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}


print ("loading data")

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

print (full_train_df.shape)


full_train_df["label_id"].to_csv("cache/full_train_df_label_ids.csv")

if __name__=="__main__":
	# unknown_df = pickle.load(open("cache/unknown_df_256.pik","rb"))
	
	X_train = []
	for i in range(11):
		exp_name = "freqconv_2xunk_gmax_onevsall_class_{}".format(i) # <<<< 
		X_train.append(np.load("cache/train_preds_{}.npy".format(exp_name)))

	X_train = np.concatenate(X_train,axis=1)

	y_train = to_categorical(full_train_df.label_id, num_classes = len(POSSIBLE_LABELS))




	X_test = []
	for i in range(11):
		exp_name = "freqconv_2xunk_gmax_onevsall_class_{}".format(i) # <<<< 
		X_test.append(np.load("cache/test_preds_{}.npy".format(exp_name)))

	X_test = np.concatenate(X_test,axis=1)



	rf = RandomForestClassifier(n_estimators=100,n_jobs=-1,class_weight='balanced')
	# mlp = MLPClassifier(100)

	rf.fit(X_train,y_train)
	# mlp.fit(X_train,y_train)

	predictions = rf.predict(X_test)
	predictions_raw  = rf.predict_proba(X_test)

	classes = np.argmax(predictions, axis=1)
	np.save("cache/classes_onevsall_1dconv_rf.npy",classes )
	np.save("cache/raw_test_preds_onevsall_1dconv_rf.npy",predictions_raw )


	test_paths = glob(os.path.join('./data/', 'test/audio/*wav'))

	### last batch will contain padding, so remove duplicates
	submission = dict()
	for i in range(len(test_paths)):
		fname, label = os.path.basename(test_paths[i]), id2name[classes[i]]
		submission[fname] = label




	with open('subm/{}.csv'.format("onevsall_1dconv_rf"), 'w') as fout: #_blend_conv1dlstm_and_aebased_conv2d_finetuned
		fout.write('fname,label\n')
		for fname, label in submission.items():
			fout.write('{},{}\n'.format(fname, label))




	# #MLP
	# predictions = mlp.predict(X_test)
	# classes = np.argmax(predictions, axis=1)
	# np.save("cache/classes_onevsall_1dconv_mlp.npy",classes )

	# ### last batch will contain padding, so remove duplicates
	# submission = dict()
	# for i in range(len(test_paths)):
	# 	fname, label = os.path.basename(test_paths[i]), id2name[classes[i]]
	# 	submission[fname] = label




	# with open('subm/{}.csv'.format("onevsall_1dconv_mlp"), 'w') as fout: #_blend_conv1dlstm_and_aebased_conv2d_finetuned
	# 	fout.write('fname,label\n')
	# 	for fname, label in submission.items():
	# 		fout.write('{},{}\n'.format(fname, label))

