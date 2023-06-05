'''
0.  Install Library
'''


!pip install tensorflow-addons==0.11.2

import tensorflow_addons as tfa

'''
1. Import Libraries for Data and Model Engineering
'''

import matplotlib.ticker as ticker
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

from tensorflow.keras.layers import Layer, Embedding, LSTM, dot, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import io

print("Tensorflow version {}".format(tf.__version__))
tf.random.set_seed(1234)
AUTO = tf.data.experimental.AUTOTUNE

'''
Part A. Data Engineering
'''

'''
2. Define Hyperparameters
'''

ENCODER_LEN = 100
DECODER_LEN = 100
BATCH_SIZE  = 128
BUFFER_SIZE = 20000

N_EPOCHS = 20

# Let's limit the #training examples for faster training
num_examples = 30000

'''
3. Import spa-eng Raw Dataset from googleapis
'''

path_to_zip = tf.keras.utils.get_file(
'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
extract=True)

file_path = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

'''
4. Define Preprocess Function
'''

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    
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

'''
5. Create Pre-processesed Datasets 
'''

def create_dataset(path, num_examples):
    # path : path to spa-eng.txt file
    # num_examples : Limit the total number of training example for faster training (set num_examples = len(lines) to use full data)
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

    return zip(*word_pairs)

# creating cleaned input, output pairs
trg_sentence, src_sentence = create_dataset(file_path, num_examples)

'''
6. Tokenizer and Vocab define
'''

# filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
# oov_token = '<unk>'

# SRC_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)
# TRG_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)

SRC_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
TRG_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')

SRC_tokenizer.fit_on_texts(src_sentence)
TRG_tokenizer.fit_on_texts(trg_sentence)

n_enc_vocab = len(SRC_tokenizer.word_index) + 1
n_dec_vocab = len(TRG_tokenizer.word_index) + 1

print('Word set size of Encoder :',n_enc_vocab)
print('Word set size of Decoder :',n_dec_vocab)

'''
7. Tokenizer test
'''

lines = [
  "Es invierno y el clima es muy frío.",
  "¿Será esta Navidad una Navidad blanca?",
  "Tenga cuidado de no resfriarse en invierno y tenga un feliz año nuevo."
]
for line in lines:
    txt_2_ids = SRC_tokenizer.texts_to_sequences([line])
    ids_2_txt = SRC_tokenizer.sequences_to_texts(txt_2_ids)
    print("Input     :", line)
    print("txt_2_ids :", txt_2_ids)
    print("ids_2_txt :", ids_2_txt[0],"\n")

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
8. Tokenize
'''
# tokenize / encode integers / add start and end tokens / padding
tokenized_inputs  = SRC_tokenizer.texts_to_sequences(src_sentence)
tokenized_outputs = TRG_tokenizer.texts_to_sequences(trg_sentence)

'''
9. Pad sequences
'''
# tkn_sources = pad_sequences(tokenized_inputs,  maxlen=ENCODER_LEN, padding='post', truncating='post')
# tkn_targets = pad_sequences(tokenized_outputs, maxlen=DECODER_LEN, padding='post', truncating='post')

tkn_sources = pad_sequences(tokenized_inputs,  padding='post')
tkn_targets = pad_sequences(tokenized_outputs, padding='post')

'''
10. [PASS] Data type define
'''

# tkn_sources = tf.cast(tkn_sources, dtype=tf.int64)
# tkn_targets = tf.cast(tkn_targets, dtype=tf.int64)

'''
11. Explore the Tokenized datasets.
'''
print('Size of source language data(shape) :', tkn_sources.shape)
print('Size of target language data(shape) :', tkn_targets.shape)

# Randomly output the 0th sample
print(tkn_sources[0])
print(tkn_targets[0])

'''
12. Split datasets.
'''

tkn_sources_train, tkn_sources_val, tkn_targets_train, tkn_targets_val = train_test_split(tkn_sources, tkn_targets, test_size=0.2)

train_dataset = tf.data.Dataset.from_tensor_slices((tkn_sources_train, tkn_targets_train))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

test_dataset   = tf.data.Dataset.from_tensor_slices((tkn_sources_val, tkn_targets_val))
test_dataset   = test_dataset.batch(BATCH_SIZE, drop_remainder=True)

print("Inpute Vocabulary Size: {}".format(len(SRC_tokenizer.word_index)))
print("Target Vocabulary Size: {}".format(len(TRG_tokenizer.word_index)))

'''
13. Define some useful parameters for further use
'''

example_input_batch, example_target_batch = next(iter(train_dataset))
example_input_batch.shape, example_target_batch.shape

max_length_input  = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

print("max_length_spanish, max_length_english, vocab_size_spanish, vocab_size_english")
max_length_input, max_length_output, n_enc_vocab, n_dec_vocab

'''
Part B. Model Engineering
'''

'''
1. [PASS] Import Libraries
'''

'''
2. Define Hyperparameters for Model Engineering
'''
embedding_dim = 128
hid_dim = 1024

steps_per_epoch = num_examples//BATCH_SIZE

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

    def call(self, inp_batch, hidden):
        inp_batch = self.embedding(inp_batch)
        output, state_h, state_c = self.lstm(inp_batch, initial_state = hidden)
        
        return output, state_h, state_c

    def initialize_hidden_state(self):
        return [tf.zeros((self.BATCH_SIZE, self.enc_units)), tf.zeros((self.BATCH_SIZE, self.enc_units))]

## Test Encoder Stack
encoder = Encoder(n_enc_vocab, embedding_dim, hid_dim, BATCH_SIZE)

# sample input
sample_hidden = encoder.initialize_hidden_state()
enc_output, sample_h, sample_c = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, hid_dim) {}'.format(enc_output.shape))
print ('Encoder h vecotr shape: (batch size, hid_dim) {}'.format(sample_h.shape))
print ('Encoder c vector shape: (batch size, hid_dim) {}'.format(sample_c.shape))

'''
5. [PASS] Build Attention Block and Exploration
'''


'''
6. Build Decoder Block and Exploration
'''
# Decoder
class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, BATCH_SIZE, attention_type='luong'):
        super(Decoder, self).__init__()
        self.BATCH_SIZE = BATCH_SIZE
        self.dec_units = dec_units
        self.attention_type = attention_type

        # Embedding Layer
        self.embedding = Embedding(vocab_size, embedding_dim)
        
        # Final Dense layer on which softmax will be applied
        self.fc = Dense(vocab_size)

        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)
        
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(self.dec_units, 
                                                                  None, self.BATCH_SIZE*[max_length_input], self.attention_type)

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = self.build_rnn_cell(BATCH_SIZE)

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)

    
    def build_rnn_cell(self, BATCH_SIZE):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, 
                                      self.attention_mechanism, attention_layer_size=self.dec_units)
        return rnn_cell

    def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
        # ------------- #
        # typ: Which sort of attention (Bahdanau, Luong)
        # dec_units: final dimension of attention outputs 
        # memory: encoder hidden states of shape (BATCH_SIZE, max_length_input, enc_units)
        # memory_sequence_length: 1d array of shape (BATCH_SIZE) with every element set to max_length_input (for masking purpose)

        if(attention_type=='bahdanau'):
            return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
        else:
            return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

    def build_initial_state(self, BATCH_SIZE, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=BATCH_SIZE, dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.BATCH_SIZE*[max_length_output-1])
        return outputs

# Test decoder stack

decoder = Decoder(n_dec_vocab, embedding_dim, hid_dim, BATCH_SIZE, 'luong')

sample_x = tf.random.uniform((BATCH_SIZE, max_length_output))
decoder.attention_mechanism.setup_memory(enc_output)
initial_state = decoder.build_initial_state(BATCH_SIZE, [sample_h, sample_c], tf.float32)


sample_decoder_outputs = decoder(sample_x, initial_state)

print("Decoder Outputs Shape: ", sample_decoder_outputs.rnn_output.shape)

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
def train_step(inp, tar, enc_hidden):
    loss = 0
    '''
    T1. Clear Gradients
    '''
    with tf.GradientTape() as tape:
        '''
        T2. Encoder Output / Hidden State, Cell State
        '''
        enc_output, enc_state_h, enc_state_c = encoder(inp, enc_hidden)
        
        '''
        T3. At the begining, set the decoder state to the encoder state
        '''
        dec_state_h, dec_state_c = enc_state_h, enc_state_c
        
        dec_input = tar[ : , :-1 ] # Ignore <end> token
        real = tar[ : , 1: ]         # ignore <start> token
        
        # Set the AttentionMechanism object with encoder_outputs
        decoder.attention_mechanism.setup_memory(enc_output)

        # Create AttentionWrapperState as initial_state for decoder
        decoder_initial_state = decoder.build_initial_state(BATCH_SIZE, [dec_state_h, dec_state_c], tf.float32)
        pred = decoder(dec_input, decoder_initial_state)
        logits = pred.rnn_output
        loss = loss_function(real, logits)
    
    '''
    T8. Compute gradients / Backpropagation
    '''
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    
    '''
    T9. Adjust learnable parameters
    '''
    optimizer.apply_gradients(zip(gradients, variables))
    
    return loss

'''
12. Epochs / each step process
'''

for epoch in range(N_EPOCHS):
    start = time.time()
    
    '''
    S1. Initialize Encoder hidden state
    '''
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    # print(enc_hidden[0].shape, enc_hidden[1].shape)
    
    for (batch, (inp, tar)) in enumerate(train_dataset.take(steps_per_epoch)):
        '''
        S2. Run training loop
        '''
        batch_loss = train_step(inp, tar, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
    '''
    S3. Checkpoint manager
    '''
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

'''
13. Explore the training result with new raw sentence
'''

def evaluate(sentence):
    '''
    E1. Data Engineering for raw text
    '''
    # Preprocess sentence
    sentence = preprocess_sentence(sentence)

    # Tokenize sentence
    inputs = [SRC_tokenizer.word_index[i] for i in sentence.split(' ')]
    
    # Padding
    inputs = pad_sequences([inputs],
                           maxlen=max_length_input,
                           padding='post')
    # Convert to tensor 
    inputs = tf.convert_to_tensor(inputs)
    inference_batch_size = inputs.shape[0]
    
    # Result Initialization
    result = ''

    '''
    E2. Initialize encoder hidden state as Zeros
    '''
    enc_start_state = [tf.zeros((inference_batch_size, hid_dim)), tf.zeros((inference_batch_size,hid_dim))]
    enc_output, enc_state_h, enc_state_c = encoder(inputs, enc_start_state)

    start_tokens = tf.fill([inference_batch_size], TRG_tokenizer.word_index['<start>'])
    end_token = TRG_tokenizer.word_index['<end>']

    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

    # Instantiate BasicDecoder object
    decoder_instance = tfa.seq2seq.BasicDecoder(cell=decoder.rnn_cell, sampler=greedy_sampler, output_layer=decoder.fc)
    # Setup Memory in decoder stack
    decoder.attention_mechanism.setup_memory(enc_output)

    # set decoder_initial_state
    decoder_initial_state = decoder.build_initial_state(inference_batch_size, [enc_state_h, enc_state_c], tf.float32)
    
    ### Since the BasicDecoder wraps around Decoder's rnn cell only, we have to ensure that the inputs to BasicDecoder 
    ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this. 
    ### We only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function
    
    decoder_embedding_matrix = decoder.embedding.variables[0]

    outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token= end_token, initial_state=decoder_initial_state)
    return outputs.sample_id.numpy()

def translate(sentence):
    result = evaluate(sentence)
    print(result)
    result = TRG_tokenizer.sequences_to_texts(result)
    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

# restoring the latest checkpoint in checkpoint_path
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

# It's really cold here.
translate(u'hace mucho frio aqui.')

# this is my life.
translate(u'esta es mi vida.')

# are you still home?
translate(u'¿todavia estan en casa?')

# wrong translation
# try to find out.
translate(u'trata de averiguarlo.')

'''
14. Beam Search
'''

def beam_evaluate(sentence, beam_width=3):
    sentence = preprocess_sentence(sentence)

    inputs = [SRC_tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                          maxlen=max_length_input,
                                                          padding='post')
    inputs = tf.convert_to_tensor(inputs)
    inference_batch_size = inputs.shape[0]
    result = ''

    enc_start_state = [tf.zeros((inference_batch_size, hid_dim)), tf.zeros((inference_batch_size,hid_dim))]
    enc_out, enc_h, enc_c = encoder(inputs, enc_start_state)

    dec_h = enc_h
    dec_c = enc_c

    start_tokens = tf.fill([inference_batch_size], TRG_tokenizer.word_index['<start>'])
    end_token = TRG_tokenizer.word_index['<end>']

    # From official documentation
    # NOTE If you are using the BeamSearchDecoder with a cell wrapped in AttentionWrapper, then you must ensure that:
    # The encoder output has been tiled to beam_width via tfa.seq2seq.tile_batch (NOT tf.tile).
    # The batch_size argument passed to the get_initial_state method of this wrapper is equal to true_batch_size * beam_width.
    # The initial state created with get_initial_state above contains a cell_state value containing properly tiled final state from the encoder.

    enc_out = tfa.seq2seq.tile_batch(enc_out, multiplier=beam_width)
    decoder.attention_mechanism.setup_memory(enc_out)
    print("beam_with * [batch_size, max_length_input, rnn_units] :  3 * [1, 16, 1024]] :", enc_out.shape)

    # set decoder_inital_state which is an AttentionWrapperState considering beam_width
    hidden_state = tfa.seq2seq.tile_batch([enc_h, enc_c], multiplier=beam_width)
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

def beam_translate(sentence):
    result, beam_scores = beam_evaluate(sentence)
    print(result.shape, beam_scores.shape)
    for beam, score in zip(result, beam_scores):
        print(beam.shape, score.shape)
        output = TRG_tokenizer.sequences_to_texts(beam)
        output = [a[:a.index('<end>')] for a in output]
        beam_score = [a.sum() for a in score]
        print('Input: %s' % (sentence))
        for i in range(len(output)):
            print('{} Predicted translation: {}  {}'.format(i+1, output[i], beam_score[i]))

beam_translate(u'hace mucho frio aqui.')

beam_translate(u'¿todavia estan en casa?')


