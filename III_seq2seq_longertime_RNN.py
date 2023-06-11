"""
This code is to perform deep learning model to analysis seasonal patterns.
Firstly I am using TensorFlow platform
"""

"""
Setting Up the Environment
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import dateutils
# Setting up to use GPU
tf.autograph.set_verbosity(0)
physical_devices = tf.config.experimental.list_physical_devices("GPU")
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class tf_Bilstm(tf.keras.Model):
    def __init__(self, vocab_size, max_seqlen, **kwargs):
        super(tf_Bilstm, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, max_seqlen
        )
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(max_seqlen)
        )
        self.dense = tf.keras.layers.Dense(64, activation="relu")
        self.out = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x):
        x = self.embedding(x)
        x = self.bilstm(x)
        x = self.dense(x)
        x = self.out(x)
        return x

class CharGenModel(tf.keras.Model):
    def __init__(self, vocab_size, num_timesteps, embedding_dim, **kwargs):
        super(CharGenModel, self).__init__(**kwargs)
        self.embbeding_layer = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim
        )
        self.rnn_layer = tf.keras.layers.GRU(
            num_timesteps,
            recurrent_initializer="glorot_uniform",
            recurrent_activation="sigmoid",
            stateful=True,
            return_sequences=True
        )
        self.dense_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        x=self.embbeding_layer(x)
        x=self.rnn_layer(x)
        x=self.dense_layer(x)
        return x

class onetoManyRNN:
    def __init__(self, X_dataset, y_dataset, num_classes=4, num_epochs=100):
        self.X_dataset = X_dataset
        self.y_dataset = y_dataset
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.model = self._model_train(learning_rate=0.001, num_epochs=self.num_epochs)

    def _generate_text(self, model, prefix_string, num_chars_to_generate=12, temperature=1.0):
        input = prefix_string.values
        input = input.tolist()
        input = tf.expand_dims(input, 0)
        text_generated = []
        model.reset_states()
        for i in range(num_chars_to_generate):
            preds = model(input)
            preds = tf.squeeze(preds, 0) / temperature
            # predict char returned by model
            pred_id = tf.random.categorical(preds, num_samples=1)[-1, 0].numpy()
            text_generated.append(pred_id)
            # pass the prediction as the next input to the model
            input = tf.expand_dims([pred_id], 0)
        return text_generated

    def _model_train(self, learning_rate=0.001, num_epochs=50):
        # setting
        sentences = self.X_dataset
        labels = self.y_dataset
        vocab_size = self.num_classes + 4
        print("vocabulary size: {:d}".format(vocab_size))
        sentences_as_ints = tf.convert_to_tensor(sentences, dtype=tf.int32)
        labels_as_ints = tf.convert_to_tensor(labels, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((sentences_as_ints, labels_as_ints))

        DATA_DIR = "dl_data"
        checkpoint_dir = os.path.join(DATA_DIR, "checkpoints/checkpointsIII")
        seq_length = 12 # 12 months is a year
        batch_size = 8
        steps_per_epoch = len(labels) // seq_length // batch_size
        dataset = dataset.shuffle(10000).batch(
            batch_size, drop_remainder=True
        )
        embedding_dim = 256

        model = CharGenModel(vocab_size, seq_length, embedding_dim)
        model.build(input_shape=(batch_size, seq_length))

        def loss(labels, predictions):
            return tf.losses.sparse_categorical_crossentropy(
                labels,
                predictions,
                from_logits=True
            )

        model.compile(optimizer=tf.optimizers.Adam(),
                      loss=loss,
                      metrics="accuracy")

        model.fit(
            dataset.repeat(),
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            # verbose=0
            # callbacks=[checkpoint_callback, tensorboard_callback]
        )
        checkpoint_file = os.path.join(
            checkpoint_dir, "III_s2s_longertime_onetoManyRNN_model"
        )
        model.save_weights(checkpoint_file)
        return model

    def predictXone(self, X_predict, X_predict_date):
        # create generative model using the trained model so far
        vocab_size = self.num_classes + 4
        seq_length = 12
        embedding_dim = 256
        DATA_DIR = "dl_data"
        checkpoint_dir = os.path.join(DATA_DIR, "checkpoints/checkpointsIII")
        checkpoint_file = os.path.join(
            checkpoint_dir, "III_s2s_longertime_onetoManyRNN_model"
        )
        gen_model = CharGenModel(vocab_size, seq_length, embedding_dim)
        gen_model.load_weights(checkpoint_file)
        gen_model.build(input_shape=(1, seq_length))
        num_dice = 11  # lucky number
        num_chars_to_generate = 12
        _date = pd.to_datetime(X_predict_date)
        # 3 months is a quarter
        diceTable_colnames = pd.to_datetime([dateutils.increment(_date, months=(i+1)) for i in range(num_chars_to_generate)]).strftime('%Y-%m')
        diceTable = pd.DataFrame(columns=diceTable_colnames)
        for d in range(num_dice):
            texts_generated = self._generate_text(gen_model, X_predict, num_chars_to_generate=num_chars_to_generate)
            diceTable.loc[d] = texts_generated
        return round(diceTable.mean(), 0).astype(np.int32)