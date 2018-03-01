from utils import *
import array 

from pydub import AudioSegment
import numba 
from multiprocessing import Pool



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
        wav = AudioSegment.from_wav(fname.replace("\\","/"))
        
        
    
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
        output = librosa.cqt(samples, sr=16000)
    else:
        output = samples
    
    
    return output



def precompute_augmentations(df,num_repeats=4):
    
    def parallelize_dataframe(df, func):
    
        df_split = np.array_split(df, 16)
        pool = Pool(16)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return df



    def create_augs(df_aug):
        df_aug['raw'] = df_aug.wav_file.apply(lambda x :  process_wav_file(x,augment=True,n_mels=256))
        
        
    df_aug= pd.concat([df]*num_repeats)        
    df_aug = parallelize_dataframe(df_aug, create_augs)    
    df = pd.concat([df, df_aug])
    
    return df 




if __name__=="__main__":
	# unknown_df = pickle.load(open("cache/unknown_df_256.pik","rb"))
	train_df = pickle.load( open("cache/train_df_256.pik","rb"))
	precompute_augmentations(train_df)


	pickle.dump( train_df,open("cache/train_df_256_aug.pik","wb"),protocol=pickle.HIGHEST_PROTOCOL)


