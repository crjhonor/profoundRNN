"""
The initiative of this code is to use more profound technic which currently used in the Recurrent Neural Network deep
learning and, try to use them upon hourly trading dataset. This dataset will not only involve log return of a target
commodity hourly trading data but will also involve a basket of commodities with highly correlations.

During the daily trading, it is essential to obtain more accurate model for hourly return of the target commodity which
should be able to guide the buy in and sell off actions. The purpose is to reduce the loss and cost coming from
volatility and uncertainty.
"""
import nltk
import mimic_seq2seq_Preproc as mms2spp
import numpy as np
import re
import shutil
import tensorflow as tf
import os
import unicodedata
import zipfile
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Obtaining dataset ====================================================================================================
indexWanted_CU0 = ['CU0', 'P0', 'Y0', 'AG0', 'BU0', 'ZN0', 'C0', 'AL0', 'RM0', 'M0', 'CF0']
indexWanted_RB0 = ['RB0', 'HC0', 'I0', 'V0', 'BU0', 'PP0', 'L0', 'FG0', 'MA0', 'SA0', 'SR0']
indexWanted_SCM = ['SCM', 'AU0', 'PG0', 'EB0', 'FU0', 'TA0', 'PP0', 'L0', 'V0', 'MA0', 'RU0']

# Include all the interested commodity indexes and making the target index as the first one.
indexList = list(np.unique(indexWanted_CU0 + indexWanted_RB0 + indexWanted_SCM))
indexWanted = ['TA0']
indexes = indexWanted.copy()
for e in indexList:
    if e != indexWanted[0]:
        indexes.append(e)

# Obtain the overall dataset.
nb_classes = 1000
rr = mms2spp.rawdataRead(indexes, nb_classes)
rawDataset, classesTable = rr.readingRawdata() # rawData should be the log return already

"""
The purpose of generating feature and label is to generate the desired form of data according the seq2seq idea of the 
RNN deep learning. The obvious try is to use every log return data of all the commodities at the same time/step t to
form the feature sequence. And that sequence is beginning with the target commodity log return at time t. The label
sequence is of the steps of trading of the target commodity beginning from time t to t + number of steps. I will try to 
generate the same length for both feature sequence and label sequence. One of the draw back of this design is, as I
have been thinking about is them the most updated features will be missing as the it will not generate a label sequence
with desired length. But, as the trading is based on hours, the fundamental logics are not changing within a week, that 
is to say the last amount of features will also be represented in the data.  
"""

sents_en, sents_fr_in, sents_fr_out, X_predict, X_predict_time = rr.download_and_read(rawDataset)

# Building up the Deep Learning Model ==================================================================================
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, num_units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(num_units)
        self.W2 = tf.keras.layers.Dense(num_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query is the decoder state at time step j
        # query.shape: (batch_size, num_units)
        # values are encoder states at every timestep i
        # values.shape: (batch_size, num_timesteps, num_units)

        # add time axis to query: (batch_size, 1, num_units)
        query_with_time_axis = tf.expand_dims(query, axis=1)
        # compute score:
        score = self.V(tf.keras.activations.tanh(
            self.W1(values) + self.W2(query_with_time_axis)))
        # compute softmax
        alignment = tf.nn.softmax(score, axis=1)
        # compute attended output
        context = tf.reduce_sum(
            tf.linalg.matmul(
                tf.linalg.matrix_transpose(alignment),
                values
            ), axis=1
        )
        context = tf.expand_dims(context, axis=1)
        return context, alignment


class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, num_units):
        super(LuongAttention, self).__init__()
        self.W = tf.keras.layers.Dense(num_units)

    def call(self, query, values):
        # add time axis to query
        query_with_time_axis = tf.expand_dims(query, axis=1)
        # compute score
        score = tf.linalg.matmul(
            query_with_time_axis, self.W(values), transpose_b=True)
        # compute softmax
        alignment = tf.nn.softmax(score, axis=2)
        # compute attended output
        context = tf.matmul(alignment, values)
        return context, alignment


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, num_timesteps,
                 embedding_dim, encoder_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.encoder_dim = encoder_dim
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, input_length=num_timesteps)
        self.rnn = tf.keras.layers.GRU(
            encoder_dim, return_sequences=True, return_state=True)

    def call(self, x, state):
        x = self.embedding(x)
        x, state = self.rnn(x, initial_state=state)
        return x, state

    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.encoder_dim))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_timesteps,
                 decoder_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.decoder_dim = decoder_dim

        # self.attention = LuongAttention(embedding_dim)
        self.attention = BahdanauAttention(embedding_dim)

        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, input_length=num_timesteps)
        self.rnn = tf.keras.layers.GRU(
            decoder_dim, return_sequences=True, return_state=True)

        self.Wc = tf.keras.layers.Dense(decoder_dim, activation="tanh")
        self.Ws = tf.keras.layers.Dense(vocab_size)

    def call(self, x, state, encoder_out):
        x = self.embedding(x)
        context, alignment = self.attention(x, encoder_out)
        x = tf.expand_dims(
            tf.concat([
                x, tf.squeeze(context, axis=1)
            ], axis=1),
            axis=1)
        x, state = self.rnn(x, state)
        x = self.Wc(x)
        x = self.Ws(x)
        return x, state, alignment


def loss_fn(ytrue, ypred):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(ytrue, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = scce(ytrue, ypred, sample_weight=mask)
    return loss


@tf.function
def train_step(encoder_in, decoder_in, decoder_out, encoder_state):
    with tf.GradientTape() as tape:
        encoder_out, encoder_state = encoder(encoder_in, encoder_state)
        decoder_state = encoder_state

        loss = 0
        for t in range(decoder_out.shape[1]):
            decoder_in_t = decoder_in[:, t]
            decoder_pred_t, decoder_state, _ = decoder(decoder_in_t,
                                                       decoder_state, encoder_out)
            loss += loss_fn(decoder_out[:, t], decoder_pred_t)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss / decoder_out.shape[1]


def predict(encoder, decoder, batch_size,
            sents_en, sents_fr_out,
            word2idx_fr, idx2word_fr):
    random_id = np.random.choice(len(sents_en))
    print("input    : ", " ".join(str(s) for s in sents_en[random_id]))
    print("label    : ", " ".join(str(s) for s in sents_fr_out[random_id]))

    encoder_in = tf.expand_dims(sents_en[random_id], axis=0)
    decoder_out = tf.expand_dims(sents_fr_out[random_id], axis=0)

    encoder_state = encoder.init_state(1)
    encoder_out, encoder_state = encoder(encoder_in, encoder_state)
    decoder_state = encoder_state

    pred_sent_fr = []
    decoder_in = tf.expand_dims(
        tf.constant(word2idx_fr["BOS"]), axis=0)

    num_predict = 0
    while True:
        decoder_pred, decoder_state, _ = decoder(
            decoder_in, decoder_state, encoder_out)
        decoder_pred = tf.argmax(decoder_pred, axis=-1)
        pred_word = idx2word_fr[decoder_pred.numpy()[0][0]]
        pred_sent_fr.append(decoder_pred)
        if pred_word == "EOS":
            break
        elif num_predict >= batch_size * 2:
            break
        num_predict += 1
        decoder_in = tf.squeeze(decoder_pred, axis=1)

    output = tf.squeeze(pred_sent_fr).numpy()
    output = output.astype(np.int32)
    print("predicted: ", " ".join(str(s) for s in output))

def evaluate_bleu_score(encoder, decoder, test_dataset,
                        word2idx_fr, idx2word_fr):
    bleu_scores = []
    smooth_fn = SmoothingFunction()

    for encoder_in, decoder_in, decoder_out in test_dataset:
        encoder_state = encoder.init_state(batch_size)
        encoder_out, encoder_state = encoder(encoder_in, encoder_state)
        decoder_state = encoder_state

        ref_sent_ids = np.zeros_like(decoder_out)
        hyp_sent_ids = np.zeros_like(decoder_out)
        for t in range(decoder_out.shape[1]):
            decoder_out_t = decoder_out[:, t]
            decoder_in_t = decoder_in[:, t]
            decoder_pred_t, decoder_state, _ = decoder(
                decoder_in_t, decoder_state, encoder_out)
            decoder_pred_t = tf.argmax(decoder_pred_t, axis=-1)
            for b in range(decoder_pred_t.shape[0]):
                ref_sent_ids[b, t] = decoder_out_t.numpy()[b]
                hyp_sent_ids[b, t] = decoder_pred_t.numpy()[b][0]

        for b in range(ref_sent_ids.shape[0]):
            ref_sent = [idx2word_fr[i] for i in ref_sent_ids[b] if i > 0]
            hyp_sent = [idx2word_fr[i] for i in hyp_sent_ids[b] if i > 0]
            # remove trailing EOS
            ref_sent = ref_sent[0:-1]
            hyp_sent = hyp_sent[0:-1]
            bleu_score = sentence_bleu([ref_sent], hyp_sent,
                                       smoothing_function=smooth_fn.method1)
            bleu_scores.append(bleu_score)

    return np.mean(np.array(bleu_scores))

def predictXone(encoder, decoder, X_predict, X_predict_time,
                batch_size, sents_en, sents_fr_out, word2idx_fr, idx2word_fr):
    """
    This function is trying to predict the target commodity's upcoming few hours close.
    """
    encoder_in = tf.expand_dims(X_predict, axis=0)
    # encoder_in = tf.expand_dims(sents_en[random_id], axis=0)
    # decoder_out = tf.expand_dims(sents_fr_out[random_id], axis=0)
    print("TIME START AT:", X_predict_time)
    print("input    : ", " ".join(str(s) for s in X_predict))

    encoder_state = encoder.init_state(1)
    encoder_out, encoder_state = encoder(encoder_in, encoder_state)
    decoder_state = encoder_state

    pred_sent_fr = []
    decoder_in = tf.expand_dims(
        tf.constant(word2idx_fr["BOS"]), axis=0)

    num_predict = 0
    while True:
        decoder_pred, decoder_state, _ = decoder(
            decoder_in, decoder_state, encoder_out)
        decoder_pred = tf.argmax(decoder_pred, axis=-1)
        pred_word = idx2word_fr[decoder_pred.numpy()[0][0]]
        pred_sent_fr.append(decoder_pred)
        if pred_word == "EOS":
            break
        elif num_predict >= batch_size * 2:
            break
        num_predict += 1
        decoder_in = tf.squeeze(decoder_pred, axis=1)

    output = tf.squeeze(pred_sent_fr).numpy()
    output = output.astype(np.int32)
    print("predicted: ", " ".join(str(s) for s in output))
    return output

EMBEDDING_DIM = 256
ENCODER_DIM, DECODER_DIM = 1024, 1024
BATCH_SIZE = 32
NUM_EPOCHS = 150

tf.random.set_seed(42)

data_dir = "./dl_data"
checkpoint_dir = os.path.join(data_dir, "checkpoints")

batch_size = BATCH_SIZE
dataset = tf.data.Dataset.from_tensor_slices((sents_en, sents_fr_in, sents_fr_out))
dataset = dataset.shuffle(10000)
test_size = dataset.__len__() // 4
test_dataset = dataset.take(test_size).batch(batch_size=batch_size, drop_remainder=True)
train_dataset = dataset.skip(test_size).batch(batch_size=batch_size, drop_remainder=True)

vocab_size_en = nb_classes + 4
vocab_size_fr = nb_classes + 4
idx2word_en = classesTable['stand for'].to_dict()
word2idx_en = {v:k for k, v in idx2word_en.items()}
idx2word_fr = idx2word_en
word2idx_fr = word2idx_en

embedding_dim = EMBEDDING_DIM
encoder_dim, decoder_dim = ENCODER_DIM, DECODER_DIM
maxlen_en = max([len(m) for m in sents_en])
maxlen_fr = max([len(m) for m in sents_fr_out])

encoder = Encoder(vocab_size_en+1, embedding_dim, maxlen_en, encoder_dim)
decoder = Decoder(vocab_size_fr+1, embedding_dim, maxlen_fr, decoder_dim)

# Test code for encoder and decoder with attention
# for encoder_in, decoder_in, decoder_out in train_dataset:
#     print("inputs:", encoder_in.shape, decoder_in.shape, decoder_out.shape)
#     encoder_state = encoder.init_state(batch_size)
#     encoder_out, encoder_state = encoder(encoder_in, encoder_state)
#     decoder_state = encoder_state
#     decoder_pred = []
#     for t in range(decoder_out.shape[1]):
#         decoder_in_t = decoder_in[:, t]
#         decoder_pred_t, decoder_state, _ = decoder(decoder_in_t,
#             decoder_state, encoder_out)
#         decoder_pred.append(decoder_pred_t.numpy())
#     decoder_pred = tf.squeeze(np.array(decoder_pred), axis=2)
#     break
# print("encoder input          :", encoder_in.shape)
# print("encoder output         :", encoder_out.shape, "state:", encoder_state.shape)
# print("decoder output (logits):", decoder_pred.shape, "state:", decoder_state.shape)
# print("decoder output (labels):", decoder_out.shape)

optimizer = tf.keras.optimizers.Adam()
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

num_epochs = NUM_EPOCHS
eval_scores = []

for e in range(num_epochs):
    encoder_state = encoder.init_state(batch_size)

    for batch, data in enumerate(train_dataset):
        encoder_in, decoder_in, decoder_out = data
        # print(encoder_in.shape, decoder_in.shape, decoder_out.shape)
        loss = train_step(
            encoder_in, decoder_in, decoder_out, encoder_state)

    print("Epoch: {}, Loss: {:.4f}".format(e + 1, loss.numpy()))

    if e % 10 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
        predict(encoder, decoder, batch_size, sents_en, sents_fr_out, word2idx_fr, idx2word_fr)
        eval_score = evaluate_bleu_score(encoder, decoder, test_dataset, word2idx_fr, idx2word_fr)
        print("Eval Score (BLEU): {:.3e}".format(eval_score))
        # eval_scores.append(eval_score)

checkpoint.save(file_prefix=checkpoint_prefix)

output = predictXone(encoder, decoder, X_predict, X_predict_time, batch_size, sents_en, sents_fr_out, word2idx_fr, idx2word_fr)

while True:
    scr_input = input("Press Enter to reload and predict again...(or \"exit\" to exit)")
    rr = mms2spp.rawdataRead(indexes, nb_classes)
    rawDataset, classesTable = rr.readingRawdata()  # rawData should be the log return already
    sents_en, sents_fr_in, sents_fr_out, X_predict, X_predict_time = rr.download_and_read(rawDataset)
    output = predictXone(encoder, decoder, X_predict, X_predict_time, batch_size, sents_en, sents_fr_out, word2idx_fr, idx2word_fr)
    if scr_input == 'exit':
        break

print("Done")