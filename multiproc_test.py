import pandas as pd
import multiprocessing as mp
from utils import *
import array 

from pydub import AudioSegment
import numba 
from multiprocessing import Pool
import dask 
import pandas as pd
import dask.dataframe as dd
from dask.multiprocessing import get

all_labels = [x[0].split('/')[-1] for x in os.walk("data/train/audio/")]
 


exclusions = ["","_background_noise_", "silence_many"]
POSSIBLE_LABELS = [item for item in all_labels if item not in exclusions]
# POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}
len(id2name)

def load_data(data_dir):
	np.random.seed = 1
	
	""" Return 2 lists of tuples:
	[(class_id, user_id, path), ...] for train
	[(class_id, user_id, path), ...] for validation
	"""
	# Just a simple regexp for paths with three groups:
	# prefix, label, user_id
#     pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
	pattern  =  re.compile("(.+[\/\\\\])?(\w+)[\/\\\\]([^-]+)-.+wav")
	all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))

	with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
		validation_files = fin.readlines()
#     print validation_files
	valset = set()
	for entry in validation_files:
		r = re.match(pattern, entry.replace("_","-"))
		if r:
			valset.add(r.group(3))
	
	possible = set(POSSIBLE_LABELS)
	
	train, val, silent, unknown = [], [],[],[]
	
	for entry in all_files:
		r = re.match(pattern, entry)
		if r:
			label, uid = r.group(2), r.group(3)
			
			if label == '_background_noise_': #we've already split up noise files into 1 seg chunks under 'silence' folder
				continue
				
			if label not in possible:
				label = 'unknown'

			label_id = name2id[label]
			sample = (label, label_id, uid, entry)
			
			if uid in valset:    
				val.append(sample)
			elif label == "silence":
				silent.append(sample)
			elif label == "unknown":
				unknown.append(sample)                

			else:
				train.append(sample)

	print('There are {} train and {} val samples'.format(len(train), len(val)))
	
	columns_list = ['label', 'label_id', 'user_id', 'wav_file']
	

	train_df = pd.DataFrame(train, columns = columns_list)
	valid_df = pd.DataFrame(val, columns = columns_list)
	silent_df = pd.DataFrame(silent, columns = columns_list)
	unknown_df = pd.DataFrame(unknown, columns = columns_list)
	
	return train_df, valid_df, unknown_df, silent_df




def augment_wav(wav,pval=0.5):
	sample_rate = 16000
	L = 1000 #16000  # 1 sec
	
#     adjust speed, with 50% chance
	wav = speed_change(wav,1.+ random.choice([.1,-0.1,0])) #random.uniform(-1, 1)*0.05) if np.random.random() < pval else wav
	
	
	#adjust volume
#     db_adjustment = random.uniform(-1, 1)*10
	wav = wav + random.choice([-10,-5,0,5,10]) #randodb_adjustment if np.random.random() < pval else wav
	 
		
	#fill to 1 second
	wav = fill_to_1sec(wav)        
		
	#shift the audio by 10 ms
	shift_length = 100
	if np.random.random() < 0.5: #shift to left
		wav = wav[:L-shift_length]+ AudioSegment.silent(shift_length,frame_rate=sample_rate)
	else: #shift to right
		wav = AudioSegment.silent(shift_length,frame_rate=sample_rate) + wav[shift_length:]
		
		
		
	#blend original file with background noise     
#     if np.random.random() < pval:
	noise = random.choice(silence_files_AS)
	db_delta = (wav.dBFS - noise.dBFS) -10.

	if db_delta< 0: #reduce intensity of loud background; if it's too silent, leave it be
		noise = noise  + db_delta
	wav = wav.overlay(noise)
 
	return wav


def process_wav_file(record, reshape=False, augment=False,pval=0.5 ,output_format='logmel',n_mels=128 ):
	
	if type(record) == str: # test files
		fname = record
		label = "test"
	else:    
		fname  = record.wav_file
		label = record.label

		
		
		
		
	if "raw_AS_wav" in record: 
		wav = record.raw_AS_wav
	else:
		fname = fname.replace("\\","/") 
		wav = AudioSegment.from_wav(fname.replace("_","-"))
		
		
	
	if (not label in ["silence"]) and augment: #no augmentation for sample files 
		wav = augment_wav(wav,pval)

	else: #make sure segment is 1 second
		wav = fill_to_1sec(wav)

		
	samples = AS_to_raw(wav)
	
	
	
	if output_format == "logmel":
		output = log_mel(samples,reshape=reshape,n_mels=n_mels)
		
	elif output_format == "mfcc":
		log_S = log_mel(samples,reshape=False,n_mels=n_mels)
		mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=40) #hirese mfcc
		delta1 = librosa.feature.delta(mfcc, order=1)#hirese mfcc
		delta2 = librosa.feature.delta(mfcc, order=2)

		output = np.stack([mfcc,delta1,delta2])
		
	elif  output_format == "cqt":   
		samples = samples/np.max(temp)
		output = librosa.cqt(samples, sr=16000 , fmin=librosa.note_to_hz('C2'),n_bins=60 * 2, bins_per_octave=12 * 2)
	else:
		output = samples
	
	
	return output


def process_frame(df):
		# process data frame
		return df.wav_file.apply(lambda x : process_wav_file(x,augment=False,n_mels=128))




if __name__=="__main__":
	train_df, valid_df, unknown_df, silent_df = load_data('./data/')
	# silence_files_AS = [AudioSegment.from_wav(x) for x in silent_df.wav_file.values]
	# filler = AudioSegment.silent(duration=1000, frame_rate = 16000)


	pool = mp.Pool(30)

	valid_df = pickle.load( open("cache/valid_df_256.pik","rb"))
	valid_df_dd = dd.from_pandas(valid_df, npartitions=4)

	# funclist = []
	# for df in np.array_split(valid_df, 4):
	# 	# process each data frame
	# 	f = pool.apply_async(process_frame,[df])
	# 	funclist.append(f)

	# results = []
	# for f in funclist:
	# 	t = f.get(timeout=120)
	# 	results.append(t)# timeout in 10 seconds
	# 	print t.shape

	# # print ("There are %d rows of data"%(result))
	# print (pd.concat(results).shape)
	res = valid_df_dd.map_partitions(lambda df: df.wav_file.apply(process_frame) ,meta=('x', 'f8')).compute(get=get) 
	print (res) 
