# https://trungtran.io/2019/03/29/neural-machine-translation-with-attention-mechanism/

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, dot, Dense
from tensorflow.data import Dataset

print("Tensorflow version {}".format(tf.__version__))
tf.random.set_seed(1234)
AUTO = tf.data.experimental.AUTOTUNE

# 1. Tokenizer Install & import
# Keras Tokenizer는 tensorflow 2.X 에서 기본으로 제공하는 tokenizer이며, word level tokenizer이다. 이는 별도의 설치가 필요 없다.

# 2. Copy or load raw data to Colab
ENCODER_LEN = 41
DECODER_LEN = ENCODER_LEN
BATCH_SIZE  = 128
BUFFER_SIZE = 20000

EPOCHS = 20

import urllib3
import zipfile
import shutil
import pandas as pd

pd.set_option('display.max_colwidth', None)

http = urllib3.PoolManager()
url ='http://www.manythings.org/anki/fra-eng.zip'
filename = 'fra-eng.zip'
path = os.getcwd()
zipfilename = os.path.join(path, filename)
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)

total_df = pd.read_csv('fra.txt', names=['SRC', 'TRG', 'lic'], sep='\t')
del total_df['lic']
print(len(total_df))

total_df = total_df.loc[:, 'SRC':'TRG']
    
total_df.head()

total_df["src_len"] = ""
total_df["trg_len"] = ""
total_df.head()

for idx in range(len(total_df['SRC'])):
    # initialize string
    text_eng = str(total_df.iloc[idx]['SRC'])

    # default separator: space
    result_eng = len(text_eng.split())
    total_df.at[idx, 'src_len'] = int(result_eng)

    text_fra = str(total_df.iloc[idx]['TRG'])
    # default separator: space
    result_fra = len(text_fra.split())
    total_df.at[idx, 'trg_len'] = int(result_fra)

print('Translation Pair :',len(total_df)) # 리뷰 개수 출력

# 3. [Optional] Delete duplicated data
total_df = total_df.drop_duplicates(subset = ["SRC"])
print('Translation Pair :',len(total_df)) # 리뷰 개수 출력

total_df = total_df.drop_duplicates(subset = ["TRG"])
print('Translation Pair :',len(total_df)) # 리뷰 개수 출력

# 4. [Optional] Select samples
# 그 결과를 새로운 변수에 할당합니다.
is_within_len = (8 < total_df['src_len']) & (total_df['src_len'] < 20) & (8 < total_df['trg_len']) & (total_df['trg_len'] < 20)
# 조건를 충족하는 데이터를 필터링하여 새로운 변수에 저장합니다.
total_df = total_df[is_within_len]

selected_df = total_df.sample(n=1024*16, # number of items from axis to return.
          random_state=1234) # seed for random number generator for reproducibility

print('Translation Pair :',len(selected_df)) # 리뷰 개수 출력

# 5. Preprocess and build list
def preprocess_eng(sentence):
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
    return sentence


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    
def preprocess_fra(sentence):
    # 위에서 구현한 함수를 내부적으로 호출
    sentence = unicode_to_ascii(sentence.lower())

    # 단어와 구두점 사이에 공백을 만듭니다.
    # Ex) "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,¿])", r" \1", sentence)

    # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환합니다.
    sentence = re.sub(r"[^a-zA-Z!.?]+", r" ", sentence)

    sentence = re.sub(r"\s+", " ", sentence)

    return sentence


# 6. Tokenizer define
selected_df['SRC'] = selected_df['SRC'].apply(preprocess_eng)
selected_df['TRG'] = selected_df['TRG'].apply(preprocess_fra)

raw_src  = selected_df['SRC']
raw_trg  = selected_df['TRG']

src_sentence  = raw_src.apply(lambda x: "<SOS> " + str(x) + " <EOS>")
dec_inputs    = raw_trg.apply(lambda x: "<SOS> "+ x)
trg_sentence  = raw_trg.apply(lambda x: x + " <EOS>")

filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
oov_token = '<unk>'

# Define tokenizer
SRC_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)
TRG_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)

SRC_tokenizer.fit_on_texts(src_sentence)
TRG_tokenizer.fit_on_texts(dec_inputs)
TRG_tokenizer.fit_on_texts(trg_sentence)

src_to_index = SRC_tokenizer.word_index
index_to_src = SRC_tokenizer.index_word

tar_to_index = TRG_tokenizer.word_index
index_to_tar = TRG_tokenizer.index_word

n_enc_vocab = len(SRC_tokenizer.word_index) + 1
n_dec_vocab = len(TRG_tokenizer.word_index) + 1

print('Size of Encoder word set :',n_enc_vocab)
print('Size of Decoder word set :',n_dec_vocab)

# 7. Tokenizer test
lines = [
  "It is winter and the weather is very cold.",
  "Will this Christmas be a white Christmas?",
  "Be careful not to catch a cold in winter and have a happy new year."
]
for line in lines:
    txt_2_ids = SRC_tokenizer.texts_to_sequences([line])
    ids_2_txt = SRC_tokenizer.sequences_to_texts(txt_2_ids)
    print("Input     :", line)
    print("txt_2_ids :", txt_2_ids)
    print("ids_2_txt :", ids_2_txt[0],"\n")

lines = [
  "C'est l'hiver et il fait très froid.",
  "Ce Noël sera-t-il un Noël blanc ?",
  "Attention à ne pas attraper froid en hiver et bonne année."
]
for line in lines:
    txt_2_ids = TRG_tokenizer.texts_to_sequences([line])
    ids_2_txt = TRG_tokenizer.sequences_to_texts(txt_2_ids)
    print("Input     :", line)
    print("txt_2_ids :", txt_2_ids)
    print("ids_2_txt :", ids_2_txt[0],"\n")

# 8. Tokenize    
# Tokenization / Integer Encoding / Adding Start Token and End Token / Padding
tokenized_inputs      = SRC_tokenizer.texts_to_sequences(src_sentence)
tokenized_dec_inputs  = TRG_tokenizer.texts_to_sequences(dec_inputs)
tokenized_outputs     = TRG_tokenizer.texts_to_sequences(trg_sentence)

max_src_len = max([len(tkn) for tkn in tokenized_inputs])
max_tar_len = max([len(tkn) for tkn in tokenized_dec_inputs])
print("Max source length :", max_src_len)
print("Max target length :", max_tar_len)

# 9. Pad sequences
encoder_input  = pad_sequences(tokenized_inputs,  maxlen=ENCODER_LEN, padding='post', truncating='post')
decoder_input  = pad_sequences(tokenized_dec_inputs, maxlen=DECODER_LEN, padding='post', truncating='post')
decoder_target = pad_sequences(tokenized_outputs, maxlen=DECODER_LEN, padding='post', truncating='post')

# 10. Data type define
# encoder_input = tf.cast(encoder_input, dtype=tf.int64)
# decoder_target = tf.cast(decoder_target, dtype=tf.int64)

# 11. Check tokenized data
# Output the 0th sample randomly
print(encoder_input[0])
print(decoder_input[0])
print(decoder_target[0])

print('Encoder Input(shape)  :', encoder_input.shape)
print('Decoder Input(shape)  :', decoder_input.shape)
print('Decoder Output(shape) :', decoder_target.shape)

embedding_dim   = 256
hidden_dim      = 1024

train_dataset = tf.data.Dataset.from_tensor_slices(
    (encoder_input, decoder_input, decoder_target))
train_dataset = train_dataset.shuffle(20).batch(BATCH_SIZE)

# Encoder
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(enc_units,
                       return_sequences=True,
                       return_state=True)

    def call(self, inp_batch, states):
        inp_batch = self.embedding(inp_batch)
        output, state_h, state_c = self.lstm(inp_batch, initial_state=states)

        return output, state_h, state_c

    def initialize_state(self, batch_size):
        return (tf.zeros([batch_size, self.enc_units]),
                tf.zeros([batch_size, self.enc_units]))
    
# Decoder
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(dec_units,
                       return_sequences=True,
                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, state):
        embed = self.embedding(sequence)
        lstm_out, state_h, state_c = self.lstm(embed, state)
        logits = self.dense(lstm_out)

        return logits, state_h, state_c

    def initialize_state(self, batch_size):
        return (tf.zeros([batch_size, self.dec_units]),
                tf.zeros([batch_size, self.dec_units]))

encoder = Encoder(n_enc_vocab, embedding_dim, hidden_dim)
decoder = Decoder(n_dec_vocab, embedding_dim, hidden_dim)

initial_states = encoder.initialize_state(1)
encoder_outputs = encoder(tf.constant([[1, 2, 3]]), initial_states)
decoder_outputs = decoder(tf.constant([[1, 2, 3]]), encoder_outputs[1:])


optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                            reduction='none')
def loss_function(real, pred):
    mask = tf.math.not_equal(real, 0)
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

@tf.function
def train_step(inp_batch, target_seq_in, target_seq_out, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        en_outputs = encoder(inp_batch, enc_hidden)
        en_states = en_outputs[1:]
        de_states = en_states

        de_outputs = decoder(target_seq_in, de_states)
        logits = de_outputs[0]
        loss = loss_function(target_seq_out, logits)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss

from tqdm import tqdm, tqdm_notebook, trange
for epoch in range(EPOCHS):

    with tqdm_notebook(total=len(train_dataset), desc=f"Train Epoch {epoch+1}") as pbar:    
        enc_hidden = encoder.initialize_state(BATCH_SIZE)
        
        train_losses = []
        # train_accuracies = []
        
        for batch, (inp, target_seq_in, target_seq_out) in enumerate(train_dataset.take(-1)):
            batch_loss = train_step(inp, target_seq_in, target_seq_out, enc_hidden)
            
            train_losses.append(batch_loss)
            # train_accuracies.append(acc)
            
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {batch_loss:.4f} ({np.mean(train_losses):.4f})")
            # pbar.set_postfix_str(f"Loss: {batch_loss:.4f} ({np.mean(train_losses):.4f}) Acc: {acc:.3f} ({np.mean(train_accuracies):.3f})")

            
    if (epoch + 1) % 5 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def evaluate_sentence(test_source_text=None):
    if test_source_text is None:
        r = np.random.choice(len(raw_encoder_input))
        test_source_text = raw_encoder_input[r]
        test_target_text = raw_data_fr[r]
    else:
        test_target_text = None
        
    test_source_seq = SRC_tokenizer.texts_to_sequences([test_source_text])
    #print(test_source_seq)

    enc_hidden = encoder.initialize_state(1)
    en_outputs = encoder(tf.constant(test_source_seq), enc_hidden)

    de_input = tf.constant([[TRG_tokenizer.word_index['<sos>']]])
    de_state_h, de_state_c = en_outputs[1:]
    out_words = []

    while True:
        de_output, de_state_h, de_state_c = decoder(
            de_input, (de_state_h, de_state_c))
        de_input = tf.argmax(de_output, -1)
        out_words.append(TRG_tokenizer.index_word[de_input.numpy()[0][0]])

        if out_words[-1] == '<eos>' or len(out_words) >= 50:
            break

    # print('>',' '.join(out_words))
    sentence = ' '.join(out_words)
    return sentence

for seq_index in [3,50,100,300,1001]: # 입력 문장의 인덱스
    
    src_text = selected_df.iloc[seq_index,0]
    trg_text = selected_df.iloc[seq_index,1]
    
    print(35 * "-")
    print("Source text  : ",src_text)
    print("Ground truth : ",trg_text)
    print("Prediction   : ",evaluate_sentence(src_text)[:-5])
    print("\n")
    