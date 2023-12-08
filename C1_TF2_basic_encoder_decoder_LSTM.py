'''
Data Engineering
'''

'''
D01. Import Libraries for Data Engineering
'''

import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata

from tqdm import tqdm, tqdm_notebook, trange

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

print("Tensorflow version {}".format(tf.__version__))
tf.random.set_seed(1234)
AUTO = tf.data.experimental.AUTOTUNE

'''
D2. Import Raw Dataset
'''

! wget http://www.manythings.org/anki/fra-eng.zip
! unzip fra-eng.zip


# 1. Tokenizer Install & import
# Keras Tokenizer는 tensorflow 2.X 에서 기본으로 제공하는 tokenizer이며, word level tokenizer이다. 이는 별도의 설치가 필요 없다.

# 2. Copy or load raw data to Colab
ENCODER_LEN  = 20
DECODER_LEN  = ENCODER_LEN
BATCH_SIZE   = 256
BUFFER_SIZE  = 20000
num_examples = 512*25

N_EPOCHS = 20

import pandas as pd

pd.set_option('display.max_colwidth', None)

train_df = pd.read_csv('fra.txt', names=['SRC', 'TRG', 'lic'], sep='\t')
del train_df['lic']
print(len(train_df))

train_df = train_df.loc[:, 'SRC':'TRG']

train_df.head()

train_df["src_len"] = ""
train_df["trg_len"] = ""
train_df.head()

# [OPT] Count the number of words
for idx in range(len(train_df['SRC'])):
    # initialize string
    text_eng = str(train_df.iloc[idx]['SRC'])

    # default separator: space
    result_eng = len(text_eng.split())
    train_df.at[idx, 'src_len'] = int(result_eng)

    text_fra = str(train_df.iloc[idx]['TRG'])
    # default separator: space
    result_fra = len(text_fra.split())
    train_df.at[idx, 'trg_len'] = int(result_fra)

print('Translation Pair :',len(train_df)) # 리뷰 개수 출력

# 3. [Optional] Delete duplicated data
train_df = train_df.drop_duplicates(subset = ["SRC"])
print('Translation Pair :',len(train_df)) # 리뷰 개수 출력

train_df = train_df.drop_duplicates(subset = ["TRG"])
print('Translation Pair :',len(train_df)) # 리뷰 개수 출력

# 4. [Optional] Select samples
# 그 결과를 새로운 변수에 할당합니다.
is_within_len = (4 < train_df['src_len']) & (train_df['src_len'] < 12) & (4 < train_df['trg_len']) & (train_df['trg_len'] < 12)
# 조건를 충족하는 데이터를 필터링하여 새로운 변수에 저장합니다.
train_df = train_df[is_within_len]

dataset_df_8096 = train_df.sample(n=num_examples, # number of items from axis to return.
          random_state=1234) # seed for random number generator for reproducibility

print('Translation Pair :',len(dataset_df_8096)) # 리뷰 개수 출력

# 5. Preprocess and build list
raw_src = []
for sentence in dataset_df_8096['SRC']:
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    raw_src.append(sentence)

raw_trg = []

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    
for sentence in dataset_df_8096['TRG']:
    # 위에서 구현한 함수를 내부적으로 호출
    sentence = unicode_to_ascii(sentence.lower())

    # 단어와 구두점 사이에 공백을 만듭니다.
    # Ex) "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,¿])", r" \1", sentence)

    # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환합니다.
    sentence = re.sub(r"[^a-zA-Z!.?]+", r" ", sentence)

    sentence = re.sub(r"\s+", " ", sentence)

    raw_trg.append(sentence)

print(raw_src[:5])
print(raw_trg[:5])

# 6. Tokenizer define for BERT. Special tokens are required.
df1 = pd.DataFrame(raw_src)
df2 = pd.DataFrame(raw_trg)

df1.rename(columns={0: "SRC"}, errors="raise", inplace=True)
df2.rename(columns={0: "TRG"}, errors="raise", inplace=True)
train_df = pd.concat([df1, df2], axis=1)

print('Translation Pair :',len(train_df)) # 리뷰 개수 출력

raw_src  = train_df['TRG']
raw_trg  = train_df['SRC']

src_add_SOS_EOS  = raw_src.apply(lambda x: "<start> " + str(x) + " <end>")
trg_add_SOS_EOS  = raw_trg.apply(lambda x: "<start> "+ x + " <end>")

src_sentence = tuple(src_add_SOS_EOS.values.tolist() )
trg_sentence = tuple(trg_add_SOS_EOS.values.tolist() )

# src_sentence = src_add_SOS_EOS.values.tolist()
# trg_sentence = trg_add_SOS_EOS.values.tolist()


'''
D10. Define tokenizer
'''

# filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
oov_token = '<unk>'

# SRC_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)
# TRG_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)

SRC_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=oov_token)
TRG_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=oov_token)

SRC_tokenizer.fit_on_texts(src_sentence)
TRG_tokenizer.fit_on_texts(trg_sentence)

src_to_index = SRC_tokenizer.word_index
index_to_src = SRC_tokenizer.index_word

tar_to_index = TRG_tokenizer.word_index
index_to_tar = TRG_tokenizer.index_word

n_enc_vocab = len(SRC_tokenizer.word_index) + 1
n_dec_vocab = len(TRG_tokenizer.word_index) + 1

print('Word set size of Encoder :',n_enc_vocab)
print('Word set size of Decoder :',n_dec_vocab)

'''
D11. Tokenizer test
'''
# Source Tokenizer
lines = [
  "C'est l'hiver et il fait très froid.",
  "Ce Noël sera-t-il un Noël blanc ?",
  "Attention à ne pas attraper froid en hiver et bonne année."
]
for line in lines:
    txt_2_ids = SRC_tokenizer.texts_to_sequences([line])
    ids_2_txt = SRC_tokenizer.sequences_to_texts(txt_2_ids)
    print("Input     :", line)
    print("txt_2_ids :", txt_2_ids)
    print("ids_2_txt :", ids_2_txt[0],"\n")

# Target Tokenizer

lines = [
  "It is winter and the weather is very cold.",
  "Will this Christmas be a white Christmas?",
  "Be careful not to catch a cold in winter and have a happy new year."
]
for line in lines:
    txt_2_ids = TRG_tokenizer.texts_to_sequences([line])
    ids_2_txt = TRG_tokenizer.sequences_to_texts(txt_2_ids)
    print("Input     :", line)
    print("txt_2_ids :", txt_2_ids)
    print("ids_2_txt :", ids_2_txt[0],"\n")

'''
D12. Tokenize
'''
# tokenize / encode integers / add start and end tokens / padding
tokenized_inputs      = SRC_tokenizer.texts_to_sequences(src_sentence)
tokenized_outputs     = TRG_tokenizer.texts_to_sequences(trg_sentence)

'''
D13. [EDA] Explore the tokenized datasets.
'''

len_result = [len(s) for s in tokenized_inputs]

print('Maximum length of source : {}'.format(np.max(len_result)))
print('Average length of source : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()


len_result = [len(s) for s in tokenized_outputs]

print('Maximum length of target : {}'.format(np.max(len_result)))
print('Average length of target : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

'''
D14. Pad sequences
'''

from tensorflow.keras.preprocessing.sequence import pad_sequences

# tkn_sources = pad_sequences(tokenized_inputs,  maxlen=ENCODER_LEN, padding='post', truncating='post')
# tkn_targets = pad_sequences(tokenized_outputs, maxlen=DECODER_LEN, padding='post', truncating='post')

tkn_sources = pad_sequences(tokenized_inputs,  padding='post')
tkn_targets = pad_sequences(tokenized_outputs, padding='post')

'''
D15. [PASS] Data type define
'''

# tkn_sources = tf.cast(tkn_sources, dtype=tf.int64)
# tkn_targets = tf.cast(tkn_targets, dtype=tf.int64)

'''
D16. [EDA] Explore the Tokenized datasets.
'''
print('Size of source language data(shape) :', tkn_sources.shape)
print('Size of target language data(shape) :', tkn_targets.shape)

# Randomly output the 0th sample
print(tkn_sources[0])
print(tkn_targets[0])

'''
D17. Split Data
'''

tkn_sources_train, tkn_sources_val, tkn_targets_train, tkn_targets_val = train_test_split(tkn_sources, tkn_targets, test_size=0.2)

'''
D18. Build dataset
'''

train_dataset = tf.data.Dataset.from_tensor_slices((tkn_sources_train, tkn_targets_train))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

test_dataset   = tf.data.Dataset.from_tensor_slices((tkn_sources_val, tkn_targets_val))
test_dataset   = test_dataset.batch(BATCH_SIZE, drop_remainder=True)

print("Inpute Vocabulary Size: {}".format(len(SRC_tokenizer.word_index)))
print("Target Vocabulary Size: {}".format(len(TRG_tokenizer.word_index)))

'''
D19. Define some useful parameters for further use
'''

example_input_batch, example_target_batch = next(iter(train_dataset))
example_input_batch.shape, example_target_batch.shape

max_length_input  = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

print("max_length_spanish : ", max_length_input)
print("max_length_english : ", max_length_output)
print("vocab_size_spanish : ", n_enc_vocab)
print("vocab_size_english : ", n_dec_vocab)

'''
Model Engineering
'''

'''
M1. Import Libraries for Model Engineering
'''
from tqdm import tqdm, tqdm_notebook, trange

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Masking

'''
M2. Define Hyperparameters for Model Engineering
'''
embedding_dim = 128
hid_dim = 1024

steps_per_epoch = num_examples//BATCH_SIZE
N_EPOCHS = 20

'''
3. [PASS] Load datasets
'''

'''
4. Build Encoder Block and Exploration
'''

## Encoder has single layer of LSTM layer on top of the embedding layer 

# Encoder
class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, BATCH_SIZE):
        super(Encoder, self).__init__()
        self.BATCH_SIZE = BATCH_SIZE
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        
        ##-------- LSTM layer in Encoder ------- ##
        self.lstm = LSTM(self.enc_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform'
                       )

    def call(self, inp_batch, state_h, state_c):
        inp_batch = self.embedding(inp_batch)
        output, state_h, state_c = self.lstm(inp_batch, initial_state=[state_h, state_c])
        
        return output, state_h, state_c

    def initialize_hidden_state(self):
        return [tf.zeros((self.BATCH_SIZE, self.enc_units)), tf.zeros((self.BATCH_SIZE, self.enc_units))]

## Test Encoder Stack
encoder = Encoder(n_enc_vocab, embedding_dim, hid_dim, BATCH_SIZE)

# sample input
[enc_state_h, enc_state_c] = encoder.initialize_hidden_state()
enc_output, sample_h, sample_c = encoder(example_input_batch, enc_state_h, enc_state_c)
print ('Encoder output shape: (batch size, sequence length, hid_dim) {}'.format(enc_output.shape))
print ('Encoder h vector shape: (batch size, hid_dim) {}'.format(sample_h.shape))
print ('Encoder c vector shape: (batch size, hid_dim) {}'.format(sample_c.shape))

'''
5. [PASS] Build Attention Block and Exploration
'''

'''
6. Build Decoder Block and Exploration
'''

# Decoder
class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, BATCH_SIZE):
        super(Decoder, self).__init__()
        self.BATCH_SIZE = BATCH_SIZE
        self.dec_units = dec_units

        # Embedding Layer
        self.embedding = Embedding(vocab_size, embedding_dim)
        
        ##--- Single LSTM layers in Decoder ----###
        self.lstm = LSTM(self.dec_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform'
                       )
        
        # Final Dense layer on which softmax will be applied
        self.fc = Dense(vocab_size)

    def call(self, dec_input, enc_h, enc_c):

        # dec_input shape after passing through embedding == (BATCH_SIZE, 1, embedding_dim)
        x = self.embedding(dec_input)
        
        #------- First call to decoder LSTM layers, pass the final hidden states of encoder to first call -----###
        output, state_h, state_c = self.lstm(x, [enc_h, enc_c])
        
        # output shape == (BATCH_SIZE * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        
        # output shape == (BATCH_SIZE, vocab)
        output = self.fc(output)
        
        return output, state_h, state_c

decoder = Decoder(n_dec_vocab, embedding_dim, hid_dim, BATCH_SIZE)

sample_decoder_output, dec_h, dec_c = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_h, sample_c)

print ('Decoder output shape: (BATCH_SIZE, vocab size) {}'.format(sample_decoder_output.shape))
print('Decoder_h shape: ', dec_h.shape)
print('Decoder_c shape: ', dec_c.shape)

'''
7. Define Loss Function
'''

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)

'''
8. Learning Rate Scheduling
'''

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, hid_dim, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.hid_dim = hid_dim
        self.hid_dim = tf.cast(self.hid_dim, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.hid_dim) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(hid_dim)

temp_learning_rate_schedule = CustomSchedule(hid_dim)

plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")

'''
9. Define Optimizer
'''
# Let's use the default parameters of Adam Optimizer

# optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
optimizer = tf.keras.optimizers.Adam()

'''
9. [Opt] Define Accuracy Metrics
Not used in this implementation.
'''

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
    
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

'''
10. [OPT] Define Checkpoints Manager
'''

checkpoint_path = "./checkpoints"

checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

'''
11. Define Training Loop
'''

@tf.function
def train_step(inp, tar, enc_state_h, enc_state_c):
    loss = 0
    '''
    T1. Clear Gradients
    '''
    with tf.GradientTape() as tape:
        '''
        T2. Encoder Output / Hidden State, Cell State
        '''
        enc_output, enc_state_h, enc_state_c = encoder(inp, enc_state_h, enc_state_c)
        
        '''
        T3. At the begining, set the decoder state to the encoder state
        '''
        dec_state_h, dec_state_c = enc_state_h, enc_state_c
        
        '''
        T4. At the begining, Feed the <start> token as Decoder-Input
        '''        
        # at the begining we feed the <start> token as input for the decoder, 
        # then we will feed the target as input
        dec_input = tf.expand_dims([TRG_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
        
        # Teacher forcing - feeding the target as the next input
        # tar.shape[1] == seq length
        for t in range(1, tar.shape[1]):
            
            '''
            T6. Decoder Output / Hidden State, Cell State
            '''
            # passing enc_output to the decoder
            predictions, dec_state_h, dec_state_c = decoder(dec_input, dec_state_h, dec_state_c)
            
            '''
            T7. Compute loss 
            '''
            loss += loss_function(tar[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(tar[:, t], 1)
        
    batch_loss = (loss / int(tar.shape[1]))
    
    '''
    T8. Compute gradients / Backpropagation
    '''
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    
    '''
    T9. Adjust learnable parameters
    '''
    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss

'''
M8. Define Episode / each step process
'''

for epoch in range(N_EPOCHS):
    start = time.time()
    
    '''
    S1. Initialize Encoder hidden state, cell state
    '''
    [enc_state_h, enc_state_c] = encoder.initialize_hidden_state()
    total_loss = 0
    
    with tqdm_notebook(total=len(train_dataset), desc=f"Train {epoch+1}") as pbar:
        for (batch, (inp, tar)) in enumerate(train_dataset.take(steps_per_epoch)):
            '''
            S2. Run training loop
            '''
            batch_loss = train_step(inp, tar, enc_state_h, enc_state_c)
            total_loss += batch_loss
            '''
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                       batch,
                                                       batch_loss.numpy()))
            '''
            pbar.update(1)
            pbar.set_postfix_str(f"Loss {batch_loss.numpy():.4f}")
            
    '''
    S3. Run Checkpoint manager
    '''
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


'''
13. Explore the training result with new raw sentence
'''
def preprocess_sentence(sentence):
    # # Call the unicode_to_ascii function implemented above internally
    sentence = unicode_to_ascii(sentence.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)

    sentence = sentence.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    sentence = '<start> ' + sentence + ' <end>'
    return sentence

def evaluate(text):
    '''
    E1. Data Engineering for raw text
    '''
    # Preprocess sentence
    sentence = preprocess_sentence(text)

    # Tokenize sentence
    inputs = [SRC_tokenizer.word_index[i] for i in sentence.split(' ')]
    
    # Padding
    inputs = pad_sequences([inputs],
                           maxlen=max_length_input,
                           padding='post')
    # Convert to tensor 
    inputs = tf.convert_to_tensor(inputs)
    
    # Result Initialization
    result = ''

    '''
    E2. Initialize encoder hidden state as Zeros
    '''
    [enc_state_h, enc_state_c] = [tf.zeros((1, hid_dim)), tf.zeros((1, hid_dim))]
    
    '''
    E3. Encoder Output / Hidden State, Cell State
    '''
    enc_out, enc_state_h, enc_state_c = encoder(inputs, enc_state_h, enc_state_c)
    
    '''
    E4. At the beginning, set the decoder state to the encoder state
    '''
    dec_state_h, dec_state_c = enc_state_h, enc_state_c
    
    '''
    E5. At the beginning, Feed the <start> token as Decoder-Input
    '''
    dec_input = tf.expand_dims([TRG_tokenizer.word_index['<start>']], 0)

    for t in range(max_length_output):
        
        '''
        E7. Decoder Output / Hidden State, Cell State
        '''
        predictions, dec_state_h, dec_state_c = decoder(dec_input, 
                                                             dec_state_h, 
                                                             dec_state_c, 
                                         )
        
        '''
        E8. Prediction ID and Build result
        '''
        predicted_id = tf.argmax(predictions[0]).numpy()
        
        '''
        E9. Stop prediction
        '''
        if TRG_tokenizer.index_word[predicted_id] == '<end>':
            return result, sentence
        
        result += TRG_tokenizer.index_word[predicted_id] + ' '

        '''
        E10. The predicted ID is fed back into the model
        '''
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence

def predict(sentence):

    result, sentence = evaluate(sentence)
    # print('Input        : %s' % (sentence))
    print('Predicted    : {}'.format(result))

    return result

# restoring the latest checkpoint in checkpoint_path
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

# It's really cold here.
# predict(u'hace mucho frio aqui.')
# print("Ground Truth : It`s really cold here. \n")


for idx in (11, 21, 31, 41, 51):
    print("Input        :", raw_src[idx])
    predict(raw_src[idx])
    print("Ground Truth :", raw_trg[idx],"\n")

'''
14. Explore the training result with test dataset
'''

def translate_batch(test_dataset):
    
    '''
    B1. Open output_text file
    '''
    with open('output_text.txt', 'w') as f:
        for (inputs, targets) in test_dataset:
            '''
            B2. Outputs Initialization
            '''
            outputs = np.zeros((BATCH_SIZE, max_length_output),dtype=np.int16)
            
            '''
            B3. Initialize encoder hidden state as Zeros
            '''
            [enc_state_h, enc_state_c] = [tf.zeros((BATCH_SIZE, hid_dim)), tf.zeros((BATCH_SIZE, hid_dim))]
            
            '''
            B4. Encoder Output / Hidden State, Cell State
            '''
            enc_output, dec_h, dec_c = encoder(inputs, enc_state_h, enc_state_c )
            
            '''
            B5. At the begining, Feed the <start> token as Decoder-Input
            '''
            dec_input = tf.expand_dims([TRG_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
            
            for t in range(max_length_output):
                '''
                B7. Decoder Output / Hidden State, Cell State
                '''
                preds, dec_h, dec_c = decoder(dec_input, dec_h, dec_c)
                
                '''
                B8. Prediction ID and Build result (Token)
                '''
                predicted_id = tf.argmax(preds, axis=1).numpy()
                outputs[:, t] = predicted_id
                
                '''
                B9. The predicted ID is fed back into the Decoder
                '''
                dec_input = tf.expand_dims(predicted_id, 1)
            '''
            B10. Tokens to Sentences
            '''
            outputs = TRG_tokenizer.sequences_to_texts(outputs)
            
            '''
            B11. Truncate the <end> and store the results
            '''
            for t, item in enumerate(outputs):
                try:
                    i = item.index('<end>')
                    f.write("%s\n" %item[:i])
                except:
                    # For those translated sequences which didn't correctly translated and have <end> token.
                    f.write("%s \n" % item)

outputs = translate_batch(test_dataset)

! head output_text.txt
! wc -l output_text.txt






