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
raw_src = []
for sentence in selected_df['SRC']:
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
    
for sentence in selected_df['TRG']:
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

# 6. Tokenizer define
df1 = pd.DataFrame(raw_src)
df2 = pd.DataFrame(raw_trg)

df1.rename(columns={0: "SRC"}, errors="raise", inplace=True)
df2.rename(columns={0: "TRG"}, errors="raise", inplace=True)
train_df = pd.concat([df1, df2], axis=1)

print('Translation Pair :',len(train_df)) # Print number of reviews
train_df.sample(10)

train_df.TRG = train_df.TRG.apply(lambda x : '\t '+ x + ' \n')
train_df.sample(10)

# 글자 집합 구축
src_vocab=set()
for line in train_df.SRC: # 1줄씩 읽음
    for char in line: # 1개의 글자씩 읽음
        src_vocab.add(char)

tar_vocab=set()
for line in train_df.TRG:
    for char in line:
        tar_vocab.add(char)

n_enc_vocab = len(src_vocab)+1
n_dec_vocab = len(tar_vocab)+1

print('Size of Encoder word set :',n_enc_vocab)
print('Size of Decoder word set :',n_dec_vocab)

src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))
print(src_vocab)
print(tar_vocab)

src_to_index = dict([(word, i+1) for i, word in enumerate(src_vocab)])
tar_to_index = dict([(word, i+1) for i, word in enumerate(tar_vocab)])

# Print Vocabulary
print("Source to Index")
print(src_to_index)
print("Target to Index")
print(tar_to_index)

encoder_input = []
for line in train_df.SRC: #입력 데이터에서 1줄씩 문장을 읽음
    temp_X = []
    for w in line: #각 줄에서 1개씩 글자를 읽음
        temp_X.append(src_to_index[w]) # 글자를 해당되는 정수로 변환
    encoder_input.append(temp_X)
print(encoder_input[:5])

decoder_input = []
for line in train_df.TRG:
    temp_X = []
    for w in line:
        temp_X.append(tar_to_index[w])
    decoder_input.append(temp_X)
print(decoder_input[:5])

decoder_target = []
for line in train_df.TRG:
    t=0
    temp_X = []
    for w in line:
        if t>0:
            temp_X.append(tar_to_index[w])
        t=t+1
    decoder_target.append(temp_X)
print(decoder_target[:5])

max_src_len = max([len(line) for line in train_df.SRC])
max_tar_len = max([len(line) for line in train_df.TRG])
print("Max source length :", max_src_len)
print("Max target length :", max_tar_len)

# 9. Pad sequences
encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')

# 10. Data type define
# encoder_input = tf.cast(encoder_input, dtype=tf.int64)
# decoder_target = tf.cast(decoder_target, dtype=tf.int64)

# 11. Check tokenized data
# Output the 0th sample randomly
print(encoder_input[0])
print(decoder_input[0])
print(decoder_target[0])

encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)

print('Encoder Input(shape)  :', encoder_input.shape)
print('Decoder Input(shape)  :', decoder_input.shape)
print('Decoder Output(shape) :', decoder_target.shape)

latent_dim = 256

encoder_inputs = Input(shape=(None, n_enc_vocab))
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
# encoder_outputs도 같이 리턴받기는 했지만 여기서는 필요없으므로 이 값은 버림.
encoder_states = [state_h, state_c]                                      # Storing the hidden state of the encoder and the cell state
# LSTM은 바닐라 RNN과는 달리 상태가 두 개. 바로 은닉 상태와 셀 상태.

decoder_inputs = Input(shape=(None, n_dec_vocab))

# return_state is True to return a state value, return_sequences is True to predict a word at all times
decoder_lstm   = LSTM(latent_dim, return_sequences=True, return_state=True) 

# Using the hidden state of the encoder as the initial hidden state (initial_state)
decoder_outputs, _, _= decoder_lstm(decoder_inputs, initial_state=encoder_states)
# 디코더의 첫 상태를 인코더의 은닉 상태, 셀 상태로 합니다.
decoder_dense = Dense(n_dec_vocab, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

model.fit(x=[encoder_input, decoder_input], y=decoder_target,
          batch_size=128, epochs=20, validation_split=0.2)

encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

encoder_model.summary()

# Start Decoder Design
# A tensor that stores the state of the previous point in time.
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# Use the state of the previous time as the initial state of the present time to predict the next word
decoder_outputs2, state_h2, state_c2 = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

decoder_states2 = [state_h2, state_c2]

# # Unlike in the training process, the hidden state and cell state of the LSTM, state_h2 state_c2, are not discarded.
decoder_outputs2 = decoder_dense(decoder_outputs2)
# Decoder Model
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs2] + decoder_states2)

index_to_src = dict((i, char) for char, i in src_to_index.items())
index_to_tar = dict((i, char) for char, i in tar_to_index.items())

def decode_sequence(input_seq):
    # 입력으로부터 인코더의 상태를 얻음
    states_value = encoder_model.predict(input_seq)

    # <SOS>에 해당하는 원-핫 벡터 생성
    target_seq = np.zeros((1, 1, n_dec_vocab))
    target_seq[0, 0, tar_to_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ""

    # stop_condition이 True가 될 때까지 루프 반복
    while not stop_condition:
        # 이점 시점의 상태 states_value를 현 시점의 초기 상태로 사용
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Convert prediction results into characters
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_to_tar[sampled_token_index]

        # Add the prediction word at the current time to the prediction sentence
        decoded_sentence += sampled_char

        # <eos>에 도달하거나 최대 길이를 넘으면 중단.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_tar_len):
            stop_condition = True

        # 현재 시점의 예측 결과를 다음 시점의 입력으로 사용하기 위해 저장
        target_seq = np.zeros((1, 1, n_dec_vocab))
        target_seq[0, 0, sampled_token_index] = 1.

        # 현재 시점의 상태를 다음 시점의 상태로 사용하기 위해 저장
        states_value = [h, c]

    return decoded_sentence

for seq_index in [3,50,100,300,1001]: # 입력 문장의 인덱스
    input_seq = encoder_input[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print(35 * "-")
    print("Source text  : ", train_df.SRC[seq_index])
    print("Ground truth : ", train_df.TRG[seq_index][1:len(train_df.TRG[seq_index])-1]) # '\t'와 '\n'을 빼고 출력
    print("Prediction   : ", decoded_sentence[:len(decoded_sentence)-1]) # '\n'을 빼고 출력
    print("\n")
