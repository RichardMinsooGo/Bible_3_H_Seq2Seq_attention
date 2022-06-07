!pip install tensorflow-addons==0.11.2
import tensorflow_addons as tfa

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, dot, Dense, LSTMCell
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

example_input_batch, example_target_batch = next(iter(train_dataset))
print("example_input_batch.shape  :", example_input_batch.shape)
print("example_target_batch.shape :", example_target_batch.shape)

max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

# Encoder
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)

        ##-------- LSTM layer in Encoder ------- ##
        self.lstm = LSTM(self.enc_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')

    def call(self, inp_batch, hidden):
        inp_batch = self.embedding(inp_batch)
        output, state_h, state_c = self.lstm(inp_batch, initial_state = hidden)
        return output, state_h, state_c

    def initialize_state(self):
        return [tf.zeros((self.batch_size, self.enc_units)), tf.zeros((self.batch_size, self.enc_units))]

## Test Encoder Stack

encoder = Encoder(n_enc_vocab, embedding_dim, hidden_dim, batch_size)

# sample input
sample_hidden = encoder.initialize_state()
sample_output, sample_h, sample_c = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder h vecotr shape: (batch size, units) {}'.format(sample_h.shape))
print ('Encoder c vector shape: (batch size, units) {}'.format(sample_c.shape))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size, attention_type='luong'):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.attention_type = attention_type
        # Embedding Layer
        self.embedding = Embedding(vocab_size, embedding_dim)
        #Final Dense layer on which softmax will be applied
        self.fc = Dense(vocab_size)
        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = LSTMCell(self.dec_units)
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(self.dec_units,
                                                                  None, self.batch_size*[max_length_input], self.attention_type)
        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = self.build_rnn_cell(batch_size)
        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)
    
    def build_rnn_cell(self, batch_size):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, 
                                      self.attention_mechanism, attention_layer_size=self.dec_units)
        return rnn_cell

    def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
        # ------------- #
        # typ: Which sort of attention (Bahdanau, Luong)
        # dec_units: final dimension of attention outputs 
        # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
        # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

        if(attention_type=='bahdanau'):
            return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
        else:
            return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

    def build_initial_state(self, batch_size, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_size, dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state


    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_size*[max_length_output-1])
        return outputs

# Test decoder stack

decoder = Decoder(n_dec_vocab, embedding_dim, hidden_dim, batch_size, 'luong')
sample_x = tf.random.uniform((batch_size, max_length_output))
decoder.attention_mechanism.setup_memory(sample_output)
initial_state = decoder.build_initial_state(batch_size, [sample_h, sample_c], tf.float32)

sample_decoder_outputs = decoder(sample_x, initial_state)

print("Decoder Outputs Shape: ", sample_decoder_outputs.rnn_output.shape)

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
        enc_output, enc_h, enc_c = encoder(inp_batch, enc_hidden)
        dec_input = targ_batch[ : , :-1 ] # Ignore <eos> token
        real = targ_batch[ : , 1: ]         # ignore <sos> token
        # Set the AttentionMechanism object with encoder_outputs
        decoder.attention_mechanism.setup_memory(enc_output)
        # Create AttentionWrapperState as initial_state for decoder
        decoder_initial_state = decoder.build_initial_state(batch_size, [enc_h, enc_c], tf.float32)
        pred = decoder(dec_input, decoder_initial_state)
        logits = pred.rnn_output
        loss = loss_function(real, logits)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss

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

def evaluate_sentence(sentence):
    # sentence = preprocess(sentence)
    inputs = [SRC_tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = pad_sequences([inputs], maxlen=max_length_input, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    inference_batch_size = inputs.shape[0]
    result = ''
    enc_start_state = [tf.zeros((inference_batch_size, hidden_dim)), tf.zeros((inference_batch_size,hidden_dim))]
    enc_out, enc_hidden, enc_c = encoder(inputs, enc_start_state)
    dec_hidden = enc_hidden
    dec_c = enc_c
    start_tokens = tf.fill([inference_batch_size], TRG_tokenizer.word_index['<sos>'])
    end_token = TRG_tokenizer.word_index['<eos>']
    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()
    # Instantiate BasicDecoder object
    decoder_instance = tfa.seq2seq.BasicDecoder(cell=decoder.rnn_cell, sampler=greedy_sampler, output_layer=decoder.fc)
    # Setup Memory in decoder stack
    decoder.attention_mechanism.setup_memory(enc_out)
    # set decoder_initial_state
    decoder_initial_state = decoder.build_initial_state(inference_batch_size, [enc_hidden, enc_c], tf.float32)

    ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder 
    ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this. 
    ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

    decoder_embedding_matrix = decoder.embedding.variables[0]
    outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token= end_token, initial_state=decoder_initial_state)
    return outputs.sample_id.numpy()

    
def translate(sentence, ground_truth=None, plot_weights=True):
    result = evaluate_sentence(sentence)
    result = TRG_tokenizer.sequences_to_texts(result)
    # print(result)
    print('Input        : %s' % (sentence))
    print('Prediction   : {}'.format(result[0]))
    if ground_truth: print(f'{"Ground truth :" :15s} {ground_truth}') 
    

def preprocess_sequence(seq, tokenizer):
    sentence = tokenizer.sequences_to_texts([seq.numpy()])[0]
    sentence = sentence.split(' ')
    sentence = [s for s in sentence if s != '<sos>' and s != '<eos>' and s != '<unk>']
    return ' '.join(sentence)

for inp_batch, targ_batch in train_dataset.take(20):
    for inp, targ in zip(inp_batch, targ_batch):
        sentence = preprocess_sequence(inp, SRC_tokenizer)
        ground_truth = preprocess_sequence(targ, TRG_tokenizer)
        translate(sentence, ground_truth)
        print()
        break

def beam_evaluate_sentence(sentence, beam_width=3):
    # sentence = preprocess(sentence)
    inputs = [SRC_tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = pad_sequences([inputs], maxlen=max_length_input, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    inference_batch_size = inputs.shape[0]
    result = ''
    enc_start_state = [tf.zeros((inference_batch_size, hidden_dim)), tf.zeros((inference_batch_size,hidden_dim))]
    enc_out, enc_hidden, enc_c = encoder(inputs, enc_start_state)
    dec_hidden = enc_hidden
    dec_c = enc_c
    start_tokens = tf.fill([inference_batch_size], TRG_tokenizer.word_index['<sos>'])
    end_token = TRG_tokenizer.word_index['<eos>']
    
    # From official documentation
    # NOTE If you are using the BeamSearchDecoder with a cell wrapped in AttentionWrapper, then you must ensure that:
    # The encoder output has been tiled to beam_width via tfa.seq2seq.tile_batch (NOT tf.tile).
    # The batch_size argument passed to the get_initial_state method of this wrapper is equal to true_batch_size * beam_width.
    # The initial state created with get_initial_state above contains a cell_state value containing properly tiled final state from the encoder.

    enc_out = tfa.seq2seq.tile_batch(enc_out, multiplier=beam_width)
    decoder.attention_mechanism.setup_memory(enc_out)
    print("beam_with * [batch_size, max_length_input, rnn_units] :  3 * [1, 16, 1024]] :", enc_out.shape)
    # set decoder_inital_state which is an AttentionWrapperState considering beam_width
    hidden_state = tfa.seq2seq.tile_batch([enc_hidden, enc_c], multiplier=beam_width)
    decoder_initial_state = decoder.rnn_cell.get_initial_state(batch_size=beam_width*inference_batch_size, dtype=tf.float32)
    decoder_initial_state = decoder_initial_state.clone(cell_state=hidden_state)

    # Instantiate BeamSearchDecoder
    decoder_instance = tfa.seq2seq.BeamSearchDecoder(decoder.rnn_cell,beam_width=beam_width, output_layer=decoder.fc)
    decoder_embedding_matrix = decoder.embedding.variables[0]
    # The BeamSearchDecoder object's call() function takes care of everything.
    outputs, final_state, sequence_lengths = decoder_instance(decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)
    # outputs is tfa.seq2seq.FinalBeamSearchDecoderOutput object. 
    # The final beam predictions are stored in outputs.predicted_id
    # outputs.beam_search_decoder_output is a tfa.seq2seq.BeamSearchDecoderOutput object which keep tracks of beam_scores and parent_ids while performing a beam decoding step
    # final_state = tfa.seq2seq.BeamSearchDecoderState object.
    # Sequence Length = [inference_batch_size, beam_width] details the maximum length of the beams that are generated

    # outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width)
    # outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width)
    # Convert the shape of outputs and beam_scores to (inference_batch_size, beam_width, time_step_outputs)
    final_outputs = tf.transpose(outputs.predicted_ids, perm=(0,2,1))
    beam_scores = tf.transpose(outputs.beam_search_decoder_output.scores, perm=(0,2,1))

    return final_outputs.numpy(), beam_scores.numpy()

def beam_translate(sentence, ground_truth):
    result, beam_scores = beam_evaluate_sentence(sentence)
    print(result.shape, beam_scores.shape)
    for beam, score in zip(result, beam_scores):
        print(beam.shape, score.shape)
        output = TRG_tokenizer.sequences_to_texts(beam)
        output = [a[:a.index('<eos>')] for a in output]
        beam_score = [a.sum() for a in score]
        print('Input          : %s' % (sentence))
        for i in range(len(output)):
            print('Prediction   {} : {}  / Beam Score :{}'.format(i+1, output[i], beam_score[i]))
    if ground_truth: print(f'{"Ground truth   :" :15s} {ground_truth}') 

for inp_batch, targ_batch in train_dataset.take(5):
    for inp, targ in zip(inp_batch, targ_batch):
        sentence = preprocess_sequence(inp, SRC_tokenizer)
        ground_truth = preprocess_sequence(targ, TRG_tokenizer)
        beam_translate(sentence, ground_truth)
        print()
        break