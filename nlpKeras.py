import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from transformers import AutoTokenizer, TFAutoModel

import numpy as np
import sklearn.model_selection
import pandas as pd

class NLPModel():
    def __init__(self, num_classes):
        self.max_length = 512
        self.language_model_name = 'bert-base-uncased'
        self.num_classes = num_classes

        # create the tokenizer, which will convert text into tokens and then
        # tokens into one-hot vectors
        tokenizer = AutoTokenizer.from_pretrained(self.language_model_name)
        self.tokenizer = tokenizer


        ############ Language Model Portion (Encoder) #################
        # grab the language model from huggingface
        language_model = TFAutoModel.from_pretrained(self.language_model_name)

        # create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        # create the embeddings - the 0th index is the last hidden layer
        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]

        # Grab the CLS token from the embeddings, which embeds the meaning of the sentence as
        # a single vector
        cls_token = embeddings[:, 0, :]

        ############  Classifier Portion (Decoder) #################
        # dense 1
        dense1 = tf.keras.layers.Dense(25, activation='gelu')
        output1 = dense1(cls_token)

        # dense 2
        dense2 = tf.keras.layers.Dense(10, activation='gelu')
        output2 = dense2(output1)

        # output_layer
        output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
        final_output = output_layer(output2)

        # combine the language model
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])

        # compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                        tfa.metrics.F1Score(self.num_classes)]
        )

    def train(self, X_train, Y_train, X_val, Y_val, epochs=100, batch_size=25):

        # create a data generator, which is responsible for tokenizing
        # the data and creating batches
        training_data_generator = DataGenerator(X_train, Y_train, batch_size, self.tokenizer, self.max_length)
        validation_data_generator = DataGenerator(X_val, Y_val, batch_size, self.tokenizer, self.max_length)

        # set up callbacks
        # NOTE: this will break with val_f1_score -- not sure why, can use custom metrics though
        callbacks = []
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                       patience=5,
                                       restore_best_weights=True,
                                       mode='min'))

        # fit the model. We pass in the data generator which will handle splitting
        # the data for each batch
        history = self.model.fit(
            training_data_generator,
            validation_data=validation_data_generator,
            epochs=epochs,
            callbacks=callbacks
         )

        return history


    def predict(self, x, batch_size=100):
            """
            Predicts labels for data
            :param x: data
            :return: predictions
            """
            tokenized = self.tokenizer(x, padding=True, truncation=True, max_length=self.max_length,
                                   return_tensors='tf')
            x = (tokenized['input_ids'], tokenized['attention_mask'])

            return self.model.predict(x, batch_size=batch_size)



class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size, tokenizer, max_length):
        self._x = x_set
        self._y = y_set
        self._batch_size = batch_size
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self):
        return int(np.ceil(len(self._x) / self._batch_size))

    def __getitem__(self, idx):
        batch_x = self._x[idx * self._batch_size:(idx + 1) * self._batch_size]
        batch_y = self._y[idx * self._batch_size:(idx + 1) * self._batch_size]

        # Tokenize the input
        tokenized = self._tokenizer(batch_x, padding=True, truncation=True, max_length=self._max_length,
                                    return_tensors='tf')

        return (tokenized['input_ids'], tokenized['attention_mask']), batch_y

    def on_epoch_end(self):
        """
        Method is called each time an epoch ends. This will shuffle the data at
        the end of an epoch, which ensures the batches are not identical each
        epoch (therefore improving performance)
        :return:
        """
        # generate shuffled indexes
        idxs = np.arange(len(self._x))
        np.random.shuffle(idxs)
        # shuffle the data
        self._x = [self._x[idx] for idx in idxs]
        self._y = self._y[idxs]


def load_data(data_file_path):
    df = pd.read_csv(data_file_path, delimiter='\t')
    labels = df['Label'].values
    labelList = []
    for i in labels:
        smallList = []
        smallList.append(int(i[1]))
        smallList.append(int(i[3]))
        smallList.append(int(i[5]))
        smallList.append(int(i[7]))
        labelList.append(smallList)
    print (type(labels[0]))
    print(labelList)
    data = df['Text'].fillna("").values.tolist()

    return data, np.array(labelList)


if __name__ == "__main__":

    # load the data
    X, Y = load_data('nlp.csv')
    num_classes = Y.shape[1]

    # perform a train/validation split. Use a seed to results are replicable(?)
    # NOTE: this is not a stratified test-train split. To make stratified, stratify=True
    #       must be specified, however that requires categorical data
    X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.2, random_state=1, shuffle=True)

    # create the model
    #model = NLPModel(num_classes)

    # train the model
    #history = model.train(X_train, Y_train, X_val, Y_val, epochs=100, batch_size=100)

    # predict with the model
    #y_pred = model.predict(X_val)
    #predicted_labels = np.identity(num_classes)[np.argmax(y_pred, axis=1)]

    pLabel = []
    for text in X_val:
        if "Is" in text or "Are" in text or "Does" in text or "Can" in text or "Could" in text or "Do" in text \
                or "yes" in text:
            pLabel.append([0, 1, 0, 0])
        elif "List" in text:
            pLabel.append([0, 0, 1, 0])
        else:
            pLabel.append([1, 0, 0, 0])


    npLabel = np.array(pLabel)
    #print(pLabel)



    # output the results
    print(sklearn.metrics.classification_report(Y_val, npLabel,
                                                target_names=['Factoid', 'Yes/No', 'List', 'Summary']))