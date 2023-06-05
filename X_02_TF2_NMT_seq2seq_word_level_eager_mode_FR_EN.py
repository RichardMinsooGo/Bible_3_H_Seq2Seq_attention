from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata

from tqdm import tqdm, tqdm_notebook, trange

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Masking

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

selected_df = total_df.sample(n=1024*10, # number of items from axis to return.
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

embedding_dim = 128
hidden_dim = 256

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb        = Embedding(n_enc_vocab, embedding_dim)(encoder_inputs)   # 임베딩 층
enc_masking    = Masking(mask_value=0.0)(enc_emb)                        # 패딩 0은 연산에서 제외
encoder_lstm   = LSTM(hidden_dim, return_state=True)                    # 상태값 리턴을 위해 return_state는 True
encoder_outputs, state_h, state_c = encoder_lstm(enc_masking)            # 은닉 상태와 셀 상태를 리턴
encoder_states = [state_h, state_c]                                      # Storing the hidden state of the encoder and the cell state

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer  = Embedding(n_dec_vocab, embedding_dim)                   # 임베딩 층
dec_emb = dec_emb_layer(decoder_inputs)                                  # 패딩 0은 연산에서 제외
dec_masking    = Masking(mask_value=0.0)(dec_emb)

# return_state is True to return a state value, return_sequences is True to predict a word at all times
decoder_lstm   = LSTM(hidden_dim, return_sequences=True, return_state=True) 

# Using the hidden state of the encoder as the initial hidden state (initial_state)
decoder_outputs, _, _ = decoder_lstm(dec_masking, initial_state=encoder_states)

# 모든 시점의 결과에 대해서 소프트맥스 함수를 사용한 출력층을 통해 단어 예측
decoder_dense = Dense(n_dec_vocab, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
model.summary()

model.fit(x=[encoder_input, decoder_input], y=decoder_target,
          batch_size=128, epochs=20, validation_split=0.2)

encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

encoder_model.summary()

# Start Decoder Design
# A tensor that stores the state of the previous point in time.
decoder_state_input_h = Input(shape=(hidden_dim,))
decoder_state_input_c = Input(shape=(hidden_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 훈련 때 사용했던 임베딩 층을 재사용
dec_emb2 = dec_emb_layer(decoder_inputs)

# Use the state of the previous time as the initial state of the present time to predict the next word
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)

decoder_states2 = [state_h2, state_c2]

# # Unlike in the training process, the hidden state and cell state of the LSTM, state_h2 state_c2, are not discarded.
decoder_outputs2 = decoder_dense(decoder_outputs2)
# Decoder Model
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs2] + decoder_states2)

def decode_sequence(input_seq):
    # 입력으로부터 인코더의 상태를 얻음
    states_value = encoder_model.predict(input_seq)

    # <SOS>에 해당하는 정수 생성
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = tar_to_index['<sos>']

    stop_condition = False
    decoded_sentence = ""

    # stop_condition이 True가 될 때까지 루프 반복
    # 구현의 간소화를 위해서 이 함수는 배치 크기를 1로 가정합니다.
    while not stop_condition:
        # 이점 시점의 상태 states_value를 현 시점의 초기 상태로 사용
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Convert prediction results into words
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_to_tar[sampled_token_index]

        # Add the prediction word at the current time to the prediction sentence
        decoded_sentence += ' '+sampled_char

        # <eos>에 도달하거나 정해진 길이를 넘으면 중단.
        if (sampled_char == '<eos>' or
           len(decoded_sentence) > max_tar_len):
            stop_condition = True

        # 현재 시점의 예측 결과를 다음 시점의 입력으로 사용하기 위해 저장
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # 현재 시점의 상태를 다음 시점의 상태로 사용하기 위해 저장
        states_value = [h, c]

    return decoded_sentence

# Convert a sequence of integers in the original text to a sequence of text
def seq2src(input_seq):
    sentence = ''
    for i in input_seq:
        if(i!=0):
            sentence = sentence + index_to_src[i]+' '
    return sentence

# Convert integer sequence of translation to text sequence
def seq2tar(input_seq):
    sentence =''
    for i in input_seq:
        if((i!=0 and i!=tar_to_index['<sos>']) and i!=tar_to_index['<eos>']):
            sentence = sentence + index_to_tar[i] + ' '
    return sentence

for seq_index in [3,50,100,300,1001]: # 입력 문장의 인덱스
    
    input_seq = encoder_input[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    
    print(35 * "-")
    print("Source text  : ",seq2src(encoder_input[seq_index]))
    print("Ground truth : ",seq2tar(decoder_input[seq_index]))
    print("Prediction   : ",decoded_sentence[1:])
    print("\n")
    
