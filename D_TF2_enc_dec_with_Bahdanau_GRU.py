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

from tensorflow.keras.layers import Layer, Embedding, GRU, Dense
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

# Encoder is composed of embedding layer and then one GRU layer. It produces outputs and last hidden states. 
# Encoder Outputs shape = (BATCH_SIZE, max_length_input, hid_dim)
# Last Hidden State Shape = (BATCH_SIZE, hid_dim)

# Encoder
class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, BATCH_SIZE):
        super(Encoder, self).__init__()
        self.BATCH_SIZE = BATCH_SIZE
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        
        ##-------- GRU layer in Encoder ------- ##
        self.gru = GRU(self.enc_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform'
                       )

    def call(self, inp_batch, hidden):
        inp_batch = self.embedding(inp_batch)
        output, state = self.gru(inp_batch, initial_state=hidden)
        
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.BATCH_SIZE, self.enc_units))

## Test Encoder Stack
encoder = Encoder(n_enc_vocab, embedding_dim, hid_dim, BATCH_SIZE)

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, hid_dim) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, hid_dim) {}'.format(sample_hidden.shape))

class BahdanauAttention(Layer):
    def __init__(self, hid_dim):
        super(BahdanauAttention, self).__init__()
        # To recall, score = V*tanh(W1(encoder_outputs) + W2(Prev Step's Hidden State))
        self.W1 = Dense(hid_dim)
        self.W2 = Dense(hid_dim)
        self.V = Dense(1)

    def call(self, query, enc_output):
        ######## query : Decoder's Last Hidden State  #####
        ######## enc_output : Encoder Outputs             #####

        # query hidden state shape == (BATCH_SIZE, hidden size)
        # query_with_time_axis shape == (BATCH_SIZE, 1, hidden size)
        # enc_output shape == (BATCH_SIZE, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (BATCH_SIZE, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (BATCH_SIZE, max_length, hid_dim)

        #-------- COMPUTING EQUATION (4)  ---------#  
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(enc_output)))

        #-------- COMPUTING EQUATION (1) ------------#
        # attention_weights shape == (BATCH_SIZE, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        
        #---------- COMPUTING EQUATION (2) -----------#
        # context_vector shape after sum == (BATCH_SIZE, hidden_size)
        context_vector = attention_weights * enc_output
        # Context vector is passed on to the curren time step's decoder cell
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, hid_dim) {}".format(attention_result.shape))
print("Attention weights shape: (BATCH_SIZE, sequence_length, 1) {}".format(attention_weights.shape))

# Decoder
class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, BATCH_SIZE):
        super(Decoder, self).__init__()
        self.BATCH_SIZE = BATCH_SIZE
        self.dec_units = dec_units

        # Embedding Layer
        self.embedding = Embedding(vocab_size, embedding_dim)
        
        ##--- Single GRU layers in Decoder ----###
        self.gru = GRU(self.dec_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform'
                       )
        
        # Final Dense layer on which softmax will be applied
        self.fc = Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, dec_input, hidden, enc_output):
        # enc_output shape == (BATCH_SIZE, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (BATCH_SIZE, 1, embedding_dim)
        x = self.embedding(dec_input)
        
        #------------ COMPUTING EQUATION (3) ----------------------#
        # x shape after concatenation == (BATCH_SIZE, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        
        # output shape == (BATCH_SIZE * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        
        # output shape == (BATCH_SIZE, vocab)
        output = self.fc(output)
        
        return output, state, attention_weights

decoder = Decoder(n_dec_vocab, embedding_dim, hid_dim, BATCH_SIZE)

sample_decoder_output, hidden_state, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (BATCH_SIZE, vocab size) {}'.format(sample_decoder_output.shape))

# Let's use the default parameters of Adam Optimizer

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)

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

# optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

optimizer = tf.keras.optimizers.Adam()

temp_learning_rate_schedule = CustomSchedule(hid_dim)

plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")


checkpoint_path = "./checkpoints"

checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

@tf.function
def train_step(inp, tar, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        # at the beginning we set the decoder state to the encoder state
        dec_hidden = enc_hidden
        
        # at the begining we feed the <start> token as input for the decoder, 
        # then we will feed the target as input
        dec_input = tf.expand_dims([TRG_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
        
        # Teacher forcing - feeding the target as the next input
        # tar.shape[1] == seq length
        for t in range(1, tar.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            
            loss += loss_function(tar[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(tar[:, t], 1)
        
    batch_loss = (loss / int(tar.shape[1]))
    
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss

for epoch in range(N_EPOCHS):
    start = time.time()
    
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    
    for (batch, (inp, tar)) in enumerate(train_dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, tar, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    inputs = [SRC_tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = pad_sequences([inputs],
                           maxlen=max_length_input,
                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, hid_dim))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    
    dec_input = tf.expand_dims([TRG_tokenizer.word_index['<start>']], 0)

    for t in range(max_length_output):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                          dec_hidden,
                                          enc_out)
        
        # storing the attention weights to plot later on

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += TRG_tokenizer.index_word[predicted_id] + ' '

        # stop prediction
        if TRG_tokenizer.index_word[predicted_id] == '<end>':
            return result, sentence

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence

def translate(sentence):
    result, sentence = evaluate(sentence)

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

def translate_batch(test_dataset):
    with open('output_text.txt', 'w') as f:
        for (inputs, targets) in test_dataset:
            outputs = np.zeros((BATCH_SIZE, max_length_output),dtype=np.int16)
            hidden_state = tf.zeros((BATCH_SIZE, hid_dim))
            enc_output, dec_h = encoder(inputs, hidden_state)
            dec_input = tf.expand_dims([TRG_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
            for t in range(max_length_output):
                preds, dec_h, _ = decoder(dec_input, dec_h, enc_output)
                predicted_id = tf.argmax(preds, axis=1).numpy()
                outputs[:, t] = predicted_id
                dec_input = tf.expand_dims(predicted_id, 1)
            outputs = TRG_tokenizer.sequences_to_texts(outputs)
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

