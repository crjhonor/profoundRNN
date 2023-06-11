"""
This code is to perform deep learning model to analysis seasonal patterns.
Firstly I am using TensorFlow platform
"""

"""
Setting Up the Environment
"""
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
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

class seasonalDeeplearning:
    def __init__(self, X_train, y_train, X_predict, indexWanted, l_classes, f_classes, y_classesTable, X_classTable):
        self.X_train = X_train
        self.y_train = y_train
        self.X_predict = X_predict
        self.indexWanted = indexWanted
        self.l_classes = l_classes
        self.f_classes = f_classes
        self.y_classesTable = y_classesTable
        self.X_classesTable = X_classTable

    def generate_text(self, model, prefix_string, num_chars_to_generate=12, temperature=1.0):
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

    def to_DataLoader(self):
        print("Done")

    def tensorflowRNN(self, learning_rate=0.001, num_epochs=200):
        """
        As the most successful implementations of RNN are within the NLP field, I am deriving my models from that field,
        and using most setting with the necessary notations.
        :param learning_rate:
        :param num_epochs:
        :param nb_classses:
        :return:
        """
        # setting
        sentences = self.X_train
        labels = self.y_train
        vocab_size = self.f_classes
        print("vocabulary size: {:d}".format(vocab_size))
        # turning the classes table into word2idx notation
        X_classesTable = self.X_classesTable.copy()
        word2idx = {}
        idx2word = {}
        for i in range(X_classesTable.shape[0]):
            cl = X_classesTable.iloc[i, ]['class']
            sd = X_classesTable.iloc[i, ]['stand for']
            word2idx[sd] = cl
            idx2word[cl] = sd
        max_seqlen = sentences.shape[1]
        # create dataset
        sentences_as_ints = tf.convert_to_tensor(sentences, dtype=tf.int32)
        labels_as_ints = tf.convert_to_tensor(labels, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((sentences_as_ints, labels_as_ints))
        # As the dataset is so small, I am not trying to separate them into train, test and validation sets, but in
        # order to smooth the programming process, I am trying to copy test directly from train, and seperate only one
        # tenth of a percent from the train using as validation.
        test_size = len(sentences) // 3
        val_size = len(sentences) // 10
        test_dataset = dataset.take(test_size)
        val_dataset = dataset.take(val_size)
        train_dataset = dataset
        batch_size = 3 # as every 3 months is a quarter
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)
        # Prepare the X_predict for prediction.
        X_predict = []
        for i in range(batch_size):
            X_predict.append(self.X_predict)
        X_predict = tf.convert_to_tensor(X_predict)

        # Building the bilstm network
        model = tf_Bilstm(vocab_size+1, max_seqlen)
        model.build(input_shape=(batch_size, max_seqlen))
        model.summary()

        # compile
        model.compile(
            loss=tf.losses.mean_squared_error,
            optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
            metrics=["accuracy"]
        )

        # training
        data_dir = "../dl_data"
        logs_dir = os.path.join("./logs")
        best_model_file = os.path.join(data_dir, "best_model.h5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(best_model_file,
                                                         save_weights_only=True,
                                                         save_best_only=True)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)
        t1_fit = time.time()
        history = model.fit(
            train_dataset, epochs=num_epochs,
            validation_data=val_dataset,
            callbacks=[checkpoint, tensorboard]
        )
        t2_fit = time.time()
        model.summary()
        print('Model Training Time Total :', (t2_fit - t1_fit), "s.")

        # Visualizing the training properties.
        # Model Error
        plt.plot(history.history['loss'], label='loss')
        # plt.plot(history.history['poisson'], label="poisson")
        plt.xlabel("Epoch")
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Prediction
        best_model = tf_Bilstm(vocab_size+1, max_seqlen)
        best_model.build(input_shape=(batch_size, max_seqlen))
        best_model.load_weights(best_model_file)
        best_model.compile(
            loss=tf.losses.mean_squared_error,
            optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
            metrics=["accuracy"]
        )
        print(self.y_classesTable)
        pred = best_model.predict(X_predict)
        classPred = np.array(np.round(pred[-1], 0))
        print('Predicted class of next month:', str(classPred))

    def onetoManyRNN(self, X_train, y_train, X_predict,
                     vocab_size=4, classesTable=None, learning_rate=0.001, num_epochs=50):
        # setting
        sentences = X_train
        labels = y_train
        vocab_size = vocab_size
        print("vocabulary size: {:d}".format(vocab_size))
        # turning the classes table into word2idx notation
        X_classesTable = classesTable
        word2idx = {}
        idx2word = {}
        for i in range(X_classesTable.shape[0]):
            cl = X_classesTable.iloc[i, ]['class']
            sd = X_classesTable.iloc[i, ]['stand for']
            word2idx[sd] = cl
            idx2word[cl] = sd
        sentences_as_ints = tf.convert_to_tensor(sentences, dtype=tf.int32)
        labels_as_ints = tf.convert_to_tensor(labels, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((sentences_as_ints, labels_as_ints))

        DATA_DIR = "../dl_data"
        CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
        seq_length = 12 # 12 months is a year
        batch_size = 3 # as 3 months is a quarter
        steps_per_epoch = len(labels)
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
            CHECKPOINT_DIR, "seasonal_rnn_{:s}".format(self.indexWanted[0])
        )
        model.save_weights(checkpoint_file)
        # create generative model using the trained model so far
        gen_model = CharGenModel(vocab_size, seq_length, embedding_dim)
        gen_model.load_weights(checkpoint_file)
        gen_model.build(input_shape=(1, seq_length))
        num_dice = 11  # lucky number
        num_chars_to_generate = 3  # 3 months is a quarter
        diceTable_colnames = ['_'.join([str(i + 1), "M+"]) for i in range(num_chars_to_generate)]
        diceTable = pd.DataFrame(columns=diceTable_colnames)
        for d in range(num_dice):
            texts_generated = self.generate_text(gen_model, X_predict, num_chars_to_generate=num_chars_to_generate)
            diceTable.loc[d] = texts_generated
        print("---")
        print(pd.DataFrame(idx2word.items()))
        print("---")
        print(diceTable)
        print("---")
        print(diceTable.mean())