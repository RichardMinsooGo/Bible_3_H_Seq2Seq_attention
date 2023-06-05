!pip install chart-studio

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Embedding, GRU, Dense
from tensorflow.data import Dataset
import string

import chart_studio.plotly
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot
#%plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go

print("Tensorflow version {}".format(tf.__version__))
tf.random.set_seed(1234)
AUTO = tf.data.experimental.AUTOTUNE

EPOCHS = 20

import urllib3
import zipfile
import shutil

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

def preprocess_eng_sentence(sentence):
    '''Function to preprocess English sentence'''
    sentence = sentence.lower() # lower casing
    sentence = re.sub("'", '', sentence) # remove the quotation marks if any
    sentence = ''.join(ch for ch in sentence if ch not in exclude)
    sentence = sentence.translate(remove_digits) # remove the digits
    sentence = sentence.strip()
    sentence = re.sub(" +", " ", sentence) # remove extra spaces
    sentence = '<sos> ' + sentence + ' <eos>' # add <sos> and <eos> tokens
    return sentence

def preprocess_port_sentence(sentence):
    '''Function to preprocess Portuguese sentence'''
    sentence = re.sub("'", '', sentence) # remove the quotation marks if any
    sentence = ''.join(ch for ch in sentence if ch not in exclude)
    sentence = sentence.strip()
    sentence = re.sub(" +", " ", sentence) # remove extra spaces
    sentence = '<sos> ' + sentence + ' <eos>' # add <sos> and <eos> tokens
    return sentence
    
lines = open(path_to_file, encoding='UTF-8').read().strip().split('\n')

# If you want a full dataset, delete below line
num_examples = 50000
start_idx = 5000
lines = lines[-(start_idx+num_examples):-start_idx]

print("total number of records: ",len(lines))

exclude = set(string.punctuation) # Set of all special characters
remove_digits = str.maketrans('', '', string.digits) # Set of all digits

# Generate pairs of cleaned English and Portuguese sentences
sent_pairs = []
for line in lines:
    sent_pair = []
    eng = line.rstrip().split('\t')[0]
    port = line.rstrip().split('\t')[1]
    eng = preprocess_eng_sentence(eng)
    sent_pair.append(eng)
    port = preprocess_port_sentence(port)
    sent_pair.append(port)
    sent_pairs.append(sent_pair)
sent_pairs[5000:5010]

# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa 
# (e.g., 5 -> "dad") for each language,
class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word_index = {}
        self.index_word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        self.word_index['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word_index[word] = index + 1

        for word, index in self.word_index.items():
            self.index_word[index] = word

# sent_pairs => already created cleaned input, output pairs
# index language using the class defined above    
SRC_tokenizer = LanguageIndex(en for en, ma in sent_pairs)
TRG_tokenizer = LanguageIndex(ma for en, ma in sent_pairs)

n_enc_vocab = len(SRC_tokenizer.word_index) + 1
n_dec_vocab = len(TRG_tokenizer.word_index) + 1

print('Size of Encoder word set :',n_enc_vocab)
print('Size of Decoder word set :',n_dec_vocab)

# 7. Tokenizer test

# 8. Tokenize    
# Tokenization / Integer Encoding / Adding Start Token and End Token / Padding

tokenized_inputs = [[SRC_tokenizer.word_index[s] for s in en.split(' ')] for en, ma in sent_pairs]

# Target sentences
tokenized_dec_inputs = [[TRG_tokenizer.word_index[s] for s in ma.split(' ')] for en, ma in sent_pairs]

def max_length(tensor):
    return max(len(t) for t in tensor)

# Calculate max_length of input and output tensor
# Here, we'll set those to the longest sentence in the dataset
max_src_len = max_length(tokenized_inputs)
max_tar_len = max_length(tokenized_dec_inputs)
print("Max source length :", max_src_len)
print("Max target length :", max_tar_len)

# Padding the input and output tensor to the maximum length
inp_tensor = pad_sequences(tokenized_inputs, maxlen=max_src_len, padding='post')
targ_tensor = pad_sequences(tokenized_dec_inputs, maxlen=max_tar_len, padding='post')

# Creating training and validation sets using an 80-20 split
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
        self.gru = GRU(self.enc_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')

    def call(self, inp_batch, hidden):
        inp_batch = self.embedding(inp_batch)
        output, state = self.gru(inp_batch, initial_state=hidden)
        return output, state

    def initialize_state(self):
        return tf.zeros((self.batch_size, self.enc_units))

## Test Encoder Stack

encoder = Encoder(n_enc_vocab, embedding_dim, hidden_dim, batch_size)

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.dec_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')
        self.fc = Dense(vocab_size)
        
        # used for attention
        self.W1 = Dense(self.dec_units)
        self.W2 = Dense(self.dec_units)
        self.V = Dense(1)
        
    def call(self, dec_input, hidden, enc_output):

        query_with_time_axis = tf.expand_dims(hidden, 1)
        
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(query_with_time_axis)))
        
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(dec_input)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        return x, state, attention_weights
        
    def initialize_state(self):
        return tf.zeros((self.batch_size, self.dec_units))

decoder = Decoder(n_dec_vocab, embedding_dim, hidden_dim, batch_size)

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
def train_step(inp_batch, targ_batch, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp_batch, enc_hidden)
        # at the beginning we set the decoder state to the encoder state
        dec_hidden = enc_hidden

        # at the begining we feed the <sos> token as input for the decoder, 
        # then we will feed the target as input
        dec_input = tf.expand_dims([TRG_tokenizer.word_index['<sos>']] * batch_size, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ_batch.shape[1]): # targ_batch.shape[1] == seq length
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

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
        enc_hidden = encoder.initialize_state()
        
        train_losses = []
        # train_accuracies = []
        
        for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            
            train_losses.append(batch_loss)
            # train_accuracies.append(acc)
            
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {batch_loss:.4f} ({np.mean(train_losses):.4f})")
            # pbar.set_postfix_str(f"Loss: {batch_loss:.4f} ({np.mean(train_losses):.4f}) Acc: {acc:.3f} ({np.mean(train_accuracies):.3f})")

            
    if (epoch + 1) % 5 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def evaluate_sentence(inputs, encoder, decoder, SRC_tokenizer, TRG_tokenizer):
    
    attention_plot = np.zeros((targ_tensor.shape[1], inp_tensor.shape[1]))
    sentence = ''
    for i in inputs[0]:
        if i == 0:
            break
        sentence = sentence + SRC_tokenizer.index_word[i] + ' '
    sentence = sentence[:-1]
    
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    hidden = [tf.zeros((1, hidden_dim))]
    enc_output, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([TRG_tokenizer.word_index['<sos>']], 0)

    for t in range(targ_tensor.shape[1]):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
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

for idx in range(5):
    ground_truth = ''
    k = np.random.randint(len(inp_tensor_val))
    random_input = inp_tensor_val[k]
    random_output = targ_tensor_val[k]
    random_input = np.expand_dims(random_input,0)
    result, sentence, attention_plot = evaluate_sentence(random_input, encoder, decoder, SRC_tokenizer, TRG_tokenizer)
    print('Input: {}'.format(sentence[8:-6]))
    print('Predicted translation: {}'.format(result[:-6]))
    for i in random_output:
        if i == 0:
            break
        ground_truth = ground_truth + TRG_tokenizer.index_word[i] + ' '
    ground_truth = ground_truth[8:-7]
    print('Actual translation: {}'.format(ground_truth))
    attention_plot = attention_plot[:len(result.split(' '))-2, 1:len(sentence.split(' '))-1]
    sentence, result = sentence.split(' '), result.split(' ')
    sentence = sentence[1:-1]
    result = result[:-2]

    # use plotly to generate the heat map
    trace = go.Heatmap(z = attention_plot, x = sentence, y = result, colorscale='greens')
    data=[trace]
    iplot(data)

