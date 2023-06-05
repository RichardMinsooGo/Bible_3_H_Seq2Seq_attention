from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, dot, Dense
from tensorflow.data import Dataset

print("Tensorflow version {}".format(tf.__version__))
tf.random.set_seed(1234)
AUTO = tf.data.experimental.AUTOTUNE

EPOCHS = 20

import urllib3
import zipfile
import shutil

from tensorflow.keras.preprocessing.text import Tokenizer

np.random.seed(42)
tf.random.set_seed(42)

http = urllib3.PoolManager()
url ='http://www.manythings.org/anki/fra-eng.zip'
filename = 'fra-eng.zip'
path = os.getcwd()
zipfilename = os.path.join(path, filename)
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)

path_to_file='/content/fra.txt'

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

# Remove the accents, clean the sentences
def preprocess(sentence):
    # 위에서 구현한 함수를 내부적으로 호출
    sentence = unicode_to_ascii(sentence.lower())

    # 단어와 구두점 사이에 공백을 만듭니다.
    # Ex) "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,¿])", r" \1", sentence)

    # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환합니다.
    sentence = re.sub(r"[^a-zA-Z!.?]+", r" ", sentence)

    sentence = re.sub(r"\s+", " ", sentence)

    # adding start-of-sequence (sos) token and end-of-sequence (eos) token
    sentence = '<sos> ' + sentence + ' <eos>'
    return sentence


lines = open(path_to_file, encoding='UTF-8').read().strip().split('\n')

# If you want a full dataset, delete below line
num_examples = 50000
start_idx = 5000
lines = lines[-(start_idx+50000):-start_idx]

# list containing word pairs in the format: [[ENGLISH], [FRENCH]]
word_pairs = [[preprocess(w) for w in l.split('\t')[:-1]] for l in lines[:num_examples]]
dec_inputs, src_sentence = zip(*word_pairs)

filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
oov_token = '<unk>'

# Define tokenizer
SRC_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)
TRG_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)

SRC_tokenizer.fit_on_texts(src_sentence)
TRG_tokenizer.fit_on_texts(dec_inputs)

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

max_src_len = max([len(tkn) for tkn in tokenized_inputs])
max_tar_len = max([len(tkn) for tkn in tokenized_dec_inputs])
print("Max source length :", max_src_len)
print("Max target length :", max_tar_len)

inp_tensor = pad_sequences(tokenized_inputs, padding='post')
targ_tensor = pad_sequences(tokenized_dec_inputs, padding='post')

# Limit the size of dataset
inp_tensor_train, inp_tensor_val, targ_tensor_train, targ_tensor_val = train_test_split(inp_tensor, targ_tensor, test_size=0.2)

# Show length
print("Input tensors  : ", inp_tensor_train.shape, inp_tensor_val.shape)
print("Target tensors : ", targ_tensor_train.shape, targ_tensor_val.shape)

def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print(f'{t} ----> {lang.index_word[t]}')

print("Input Language; Index to Word Mapping")
convert(SRC_tokenizer, inp_tensor_train[0])
print()
print("Target Language; Index to Word Mapping")
convert(TRG_tokenizer, targ_tensor_train[0])

# Creating a tf.data Dataset
buffer_size     = len(inp_tensor_train)
batch_size      = 64
steps_per_epoch = len(inp_tensor_train) // batch_size
embedding_dim   = 256
hidden_dim      = 1024

"""
train_dataset = Dataset.from_tensor_slices((inp_tensor_train, targ_tensor_train)).shuffle(buffer_size)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
"""

def create_dataset(shuffle=True, buffer_size=buffer_size, batch_size=batch_size):
    ds = tf.data.Dataset.from_tensor_slices((inp_tensor_train, targ_tensor_train))
    if shuffle:
        ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(1)

train_dataset = create_dataset()
valid_dataset = create_dataset(shuffle=False)

inp_batch, targ_batch = next(iter(train_dataset))

print("inp_batch.shape  : ", inp_batch.shape)
print("targ_batch.shape : ", targ_batch.shape)

# Encoder
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(self.enc_units, return_state=True, return_sequences=True)

    def call(self, inp_batch, state_h, state_c):
        inp_batch = self.embedding(inp_batch)
        output, state_h, statec = self.lstm(inp_batch, initial_state=[state_h, state_c])
        return output, state_h, state_c

    def initialize_state(self):
        return [tf.zeros((self.batch_size, self.enc_units)), tf.zeros((self.batch_size, self.enc_units))]

## Test Encoder Stack

encoder = Encoder(n_enc_vocab, embedding_dim, hidden_dim, batch_size)

# sample input
[enc_state_h, enc_state_c] = encoder.initialize_state()
enc_output, enc_state_h, enc_state_c = encoder(inp_batch, enc_state_h, enc_state_c)

print (f'encoder output:       (batch size, seq length, enc_units)  {enc_output.shape}')
print (f'encoder hidden state h: (batch size, enc_units)            {enc_state_h.shape}')
print (f'encoder hidden state c: (batch size, enc_units)            {enc_state_c.shape}')

def print_shapes(enc_output, dec_state, score, attention_weights, context_vector):
    print(f"btach_size: {batch_size}")
    print(f"seq_length: {inp_tensor_train.shape[1]}")
    print(f"enc_units: {hidden_dim}")
    print()
    print(f"enc_output:        {enc_output.shape}")
    print(f"dec_state:         {dec_state.shape}")
    print(f"score:             {score.shape}")
    print(f"attention_weights: {attention_weights.shape}")
    print(f"context_vector:    {context_vector.shape}")

class LuongAttention(tf.keras.layers.Layer):
    def __init(self):
        super().__init__()

    def call(self, dec_state_h, dec_state_c, enc_output, prints=False):
        dec_state = tf.add(dec_state_h, dec_state_c)
        dec_state = dec_state[:, :, tf.newaxis]
    
        score = dot([enc_output, dec_state], axes=[2, 1])
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        if prints:
            print_shapes(enc_output, dec_state, score, attention_weights, context_vector)
            return
        return context_vector, attention_weights

attention_layer = LuongAttention()
# at the beginning we set the decoder state to the encoder state
dec_state_h, dec_state_c = enc_state_h, enc_state_c


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(dec_units, return_state=True, return_sequences=True)
        self.fc = Dense(vocab_size, activation="softmax")
        self.attention = LuongAttention()

    def call(self, dec_input, dec_state_h, dec_state_c, enc_output):
        context_vector, attention_weights = self.attention(dec_state_h, dec_state_c, enc_output)
        # context_vactor: (batch_size, 1, embedding_dim)
        context_vector = context_vector[:, tf.newaxis, :]

        # x: (batch_size, 1, embedding_dim)
        x = self.embedding(dec_input)
        # x: (batch_size, 1, embedding_dim + enc_units)
        x = tf.concat([context_vector, x], axis=-1)
        
        # output: (batch_size, 1, dec_units), state: (batch_size, dec_units)
        output, state_h, state_c = self.lstm(x)
        # output: (batch_size, dec_units)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state_h, state_c, attention_weights

decoder = Decoder(n_dec_vocab, embedding_dim, hidden_dim, batch_size)
dec_input = tf.random.uniform((batch_size, 1))
dec_output, dec_state_h, dec_state_c, _ = decoder(dec_input, dec_state_h, dec_state_c, enc_output)

print (f'decoder output:       (batch size, vocab_size)  {dec_output.shape}')
print (f'decoder hidden state h: (batch size, dec_units)   {dec_state_h.shape}')
print (f'decoder hidden state c: (batch size, dec_units)   {dec_state_c.shape}')

optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                                            reduction='none')
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
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
def train_step(inp_batch, targ_batch, enc_state_h, enc_state_c):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_state_h, enc_state_c = encoder(inp_batch, enc_state_h, enc_state_c)
        # at the beginning we set the decoder state to the encoder state
        dec_state_h, dec_state_c = enc_state_h, enc_state_c

        # at the begining we feed the <sos> token as input for the decoder, 
        # then we will feed the target as input
        dec_input = tf.expand_dims([TRG_tokenizer.word_index['<sos>']] * batch_size, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ_batch.shape[1]): # targ_batch.shape[1] == seq length
            # passing enc_output to the decoder
            predictions, dec_state_h, dec_state_c, _ = decoder(dec_input, dec_state_h, dec_state_c, enc_output)

            loss += loss_function(targ_batch[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ_batch[:, t], 1)
        
    batch_loss = loss / int(targ_batch.shape[1])
    
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

from tqdm import tqdm, tqdm_notebook, trange
for epoch in range(EPOCHS):

    with tqdm_notebook(total=len(train_dataset), desc=f"Train Epoch {epoch+1}") as pbar:    
        [enc_state_h, enc_state_c] = encoder.initialize_state()
        
        train_losses = []
        # train_accuracies = []
        
        for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_state_h, enc_state_c)
            
            train_losses.append(batch_loss)
            # train_accuracies.append(acc)
            
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {batch_loss:.4f} ({np.mean(train_losses):.4f})")
            # pbar.set_postfix_str(f"Loss: {batch_loss:.4f} ({np.mean(train_losses):.4f}) Acc: {acc:.3f} ({np.mean(train_accuracies):.3f})")

            
    if (epoch + 1) % 5 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def evaluate(sentence, targ_tensor, inp_tensor):
    # targ_tensor.shape[1] == max seq length for the target language (EN)
    # inp_tensor.shape[1] == max seq length for the input language (FR)
    attention_plot = np.zeros((targ_tensor.shape[1], inp_tensor.shape[1]))
    sentence = preprocess(sentence)

    inputs = SRC_tokenizer.texts_to_sequences([sentence])
    inputs = pad_sequences(inputs, maxlen=inp_tensor.shape[1], padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    [enc_state_h, enc_state_c] = [tf.zeros((1, hidden_dim)), tf.zeros((1, hidden_dim))]
    enc_output, enc_state_h, enc_state_c = encoder(inputs, enc_state_h, enc_state_c)
    dec_state_h, dec_state_c = enc_state_h, enc_state_c
    dec_input = tf.expand_dims([TRG_tokenizer.word_index['<sos>']], 0)

    for t in range(targ_tensor.shape[1]):
        predictions, dec_state_h, dec_state_c, attention_weights = decoder(dec_input, 
                                                             dec_state_h, 
                                                             dec_state_c, 
                                                             enc_output)
        
        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result += TRG_tokenizer.index_word[predicted_id] + ' '

        # stop prediction
        if TRG_tokenizer.index_word[predicted_id] == '<eos>':
            return result, sentence, attention_plot
        
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

# function for plotting the attention weights to visualize how the model works internally
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    ax.set_xticklabels([''] + sentence, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence)

    plt.show()

def translate(sentence, ground_truth=None, plot_weights=True):
    result, sentence, attention_plot = evaluate(sentence, targ_tensor, inp_tensor)

    print(f'{"Input        :":15s} {sentence}')
    print(f'{"Prediction   :":15s} {result}')
    if ground_truth: print(f'{"Ground truth :" :15s} {ground_truth}') 
    
    if plot_weights:
        attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
        plot_attention(attention_plot, sentence.split(' '), result.split(' '))

def preprocess_sequence(seq, tokenizer):
    sentence = tokenizer.sequences_to_texts([seq.numpy()])[0]
    sentence = sentence.split(' ')
    sentence = [s for s in sentence if s != '<sos>' and s != '<eos>' and s != '<unk>']
    return ' '.join(sentence)

for inp_batch, targ_batch in train_dataset.take(20):
    for inp, targ in zip(inp_batch, targ_batch):
        sentence = preprocess_sequence(inp, SRC_tokenizer)
        ground_truth = preprocess_sequence(targ, TRG_tokenizer)
        translate(sentence, ground_truth, plot_weights=False)
        print()
        break
