#!/home/go96bix/my/programs/Python-3.6.1/bin/python3.6
from pyexpat import model

import numpy as np
# import datetime
import tensorflow as tf
from keras import backend as K
import pandas as pd
from keras.utils import multi_gpu_model
from sklearn.utils import class_weight as clw
# from keras.backend.cntk_backend import argmax
# from keras.backend.cntk_backend import dropout
# from keras.wrappers.scikit_learn import KerasClassifier
# from matplotlib.dates import num2date
# from nbformat.v1 import nbbase
# from pandas.util._decorators import docstring_wrapper
# from scipy.special.basic import bessel_diff_formula
# from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import SimpleRNN, LSTM, Dense, Flatten, Embedding, Dropout, GRU, CuDNNLSTM, BatchNormalization, \
    Bidirectional, Conv1D, MaxPooling1D, Input, Concatenate
from keras.models import load_model
from keras import optimizers
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
import keras
# from sklearn.model_selection import KFold, StratifiedKFold
# from sklearn.model_selection import cross_val_score
import uuid
from keras.callbacks import ModelCheckpoint
import re
import os
# from hyperopt import Trials, STATUS_OK, tpe
# from hyperas import optim
# from hyperas.distributions import choice, uniform, conditional
from snapshot import SnapshotCallbackBuilder
import sklearn.metrics as metrics
from sklearn.metrics import log_loss
from scipy.optimize import minimize
import scipy.stats
import time
import matplotlib

matplotlib.rcParams['backend'] = 'Agg'
import matplotlib.pyplot as plt
import plotting_results
import random

# import matplotlib.mlab as mlab
# import matplotlib.patches as mpatches

"""limit number cores"""


# numCores = 15
# config = tf.ConfigProto(intra_op_parallelism_threads=numCores, inter_op_parallelism_threads=numCores, \
#                         allow_soft_placement=True, device_count = {'CPU': 1})
# session = tf.Session(config=config)
# K.set_session(session)


# def scheduler(epoch):
#     """
#     https://github.com/fchollet/keras/issues/898
#     """
#     if epoch%2==0 and epoch!=0:
#         lr = K.get_value(model.optimizer.lr)
#         K.set_value(model.optimizer.lr, lr*.9)
#         print("lr changed to {}".format(lr*.9))
#     return K.get_value(model.optimizer.lr)

class lrManipulator(keras.callbacks.Callback):
    """
    Manipulate the lr for Adam Optimizer
    -> no big chances usefull
    """

    def __init__(self, nb_epochs, nb_snapshots):
        self.T = nb_epochs
        self.M = nb_snapshots

    def on_epoch_begin(self, epoch, logs={}):
        K.set_value(self.model.optimizer.lr, 0.001)
        if ((epoch % (self.T // self.M)) == 0):
            K.set_value(self.model.optimizer.iterations, 0)
            K.set_value(self.model.optimizer.lr, 0.01)

            # um modell schnell zufaellig zu inititialisiseren, tausche gewischte der Knoten
            # weights = self.model.get_weights()
            # weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
            # self.model.set_weights(weights)


class TimeHistory(keras.callbacks.Callback):
    """https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit"""

    def on_train_begin(self, logs={}):
        self.times = []
        self.time_train_start = time.time()

    # def on_epoch_begin(self, batch, logs={}):
    #     self.epoch_time_start = datetime.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(int(time.time()) - int(self.time_train_start))


class accuracyHistory(keras.callbacks.Callback):
    """to get the accuracy of my personal voting scores"""

    def on_train_begin(self, logs={}):
        self.meanVote_train = []
        self.normalVote_train = []

        self.meanVote_val = []
        self.normalVote_val = []

        # self.meanVote_test = []
        # self.normalVote_test = []

    # def on_epoch_begin(self, batch, logs={}):
    #     self.epoch_time_start = datetime.time()

    def on_epoch_end(self, batch, logs={}):
        """
        1. make prediction of train
        2. get the voting results
        3. calc and save accuracy
        4. do same for test set
        """
        self.prediction_train = (self.model.predict(X_train))
        y_true_small, y_pred_mean_train, y_pred_voted_train, y_pred, y_pred_mean_exact = \
            calc_predictions(X_train, Y_train, do_print=False, y_pred=self.prediction_train)
        self.normalVote_train.append(metrics.accuracy_score(y_true_small, y_pred_voted_train))
        self.meanVote_train.append(metrics.accuracy_score(y_true_small, y_pred_mean_train))

        self.prediction_val = (self.model.predict(X_val))
        y_true_small, y_pred_mean_val, y_pred_voted_val, y_pred, y_pred_mean_exact = \
            calc_predictions(X_val, Y_val, do_print=False, y_pred=self.prediction_val)
        self.normalVote_val.append(metrics.accuracy_score(y_true_small, y_pred_voted_val))
        self.meanVote_val.append(metrics.accuracy_score(y_true_small, y_pred_mean_val))

    # self.prediction_test = (self.model.predict(X_test))
    # y_true_small, y_pred_mean_test, y_pred_voted_test, y_pred, y_pred_mean_exact = \
    #     calc_predictions(X_test, Y_test, do_print=False, y_pred=self.prediction_test)
    # self.normalVote_test.append(metrics.accuracy_score(y_true_small, y_pred_voted_test))
    # self.meanVote_test.append(metrics.accuracy_score(y_true_small, y_pred_mean_test))


class prediction_history(keras.callbacks.Callback):
    """Callback subclass that prints each epoch prediction"""

    def on_epoch_end(self, epoch, logs={}):
        print()
        print(K.get_value(self.model.optimizer.lr))
        print(K.get_value(self.model.optimizer.iterations))
        # shuffle X and Y in sam way
        p = np.random.permutation(len(Y_test))
        shuffled_X = X_test[p]
        shuffled_Y = Y_test[p]
        self.predhis = (self.model.predict(shuffled_X[0:10]))
        # y_pred = my_model.predict(X_test)
        # print(self.predhis)
        y_pred = np.argmax(self.predhis, axis=-1)
        y_true = np.argmax(shuffled_Y, axis=-1)[0:10]
        print(y_pred)
        print(y_true)
        table = pd.crosstab(
            pd.Series(y_true),
            pd.Series(y_pred),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print(table)


def shrink_timesteps(input_subSeqlength=0):
    """
        needed for Truncated Backpropagation Through Time
    If you have long input sequences, such as thousands of timesteps,
    you may need to break the long input sequences into multiple contiguous subsequences.

    e.g. 100 subseq.
    Care would be needed to preserve state across each 100 subsequences and reset
    the internal state after each 100 samples either explicitly or by using a batch size of 100.
    :param input_subSeqlength: set for specific subsequence length
    :return:
    """
    global X_train, X_test, X_val, Y_train, Y_test, Y_val, batch_size
    print(X_train.max())
    samples = X_train.shape[0]
    seqlength = X_train.shape[1]
    features = X_train.shape[2]

    # search for possible new shape without loss
    subSeqlength = input_subSeqlength
    if input_subSeqlength == 0:
        for i in range(200, 400):
            if (seqlength % i == 0):
                subSeqlength = i

                X_train = X_train.reshape((int(seqlength / subSeqlength) * samples, subSeqlength, features))
                X_test = X_test.reshape((int(seqlength / subSeqlength) * X_test.shape[0], subSeqlength, features))
                X_val = X_val.reshape((int(seqlength / subSeqlength) * X_val.shape[0], subSeqlength, features))
                break
    # if without loss not possible or special shape wanted
    if input_subSeqlength == subSeqlength:
        if subSeqlength == 0:
            subSeqlength = 100
        # cut of the end of the sequence
        newSeqlength = int(seqlength / subSeqlength) * subSeqlength
        i = 0
        for j in (X_train, X_test, X_val):
            bigarray = []
            for sample in j:
                sample = np.array(sample[0:newSeqlength], dtype=int)
                subarray = sample.reshape((int(seqlength / subSeqlength), subSeqlength, features))
                bigarray.append(subarray)
            bigarray = np.array(bigarray)
            bigarray = bigarray.reshape((bigarray.shape[0] * bigarray.shape[1], bigarray.shape[2], bigarray.shape[3]))
            if i == 0:
                X_train = bigarray
            if i == 1:
                X_test = bigarray
            else:
                X_val = bigarray
            i += 1
    # expand Y files (real classes for samples)
    i = 0
    for j in (Y_train, Y_test, Y_val):
        bigarray = []
        for sample in j:
            bigarray.append(int(seqlength / subSeqlength) * [sample])
        bigarray = np.array(bigarray)
        bigarray = bigarray.reshape((bigarray.shape[0] * bigarray.shape[1], bigarray.shape[2]))
        if (i == 0):
            Y_train = bigarray
        if i == 1:
            Y_test = bigarray
        else:
            Y_val = bigarray
        i += 1
    batch_size = int(seqlength / subSeqlength)


def use_old_data_max(one_hot_encoding=True):
    """
    to reuse the "old" exported data
    """

    Y_train_old = np.genfromtxt(directory + '/Y_train.csv', delimiter=',', dtype='int16')
    Y_test_old = np.genfromtxt(directory + '/Y_test.csv', delimiter=',', dtype='int16')
    X_train_old = np.genfromtxt(directory + '/X_train.csv', delimiter=',', dtype='str')
    X_test_old = np.genfromtxt(directory + '/X_test.csv', delimiter=',', dtype='str')

    def one_hot_encode_int(data):

        """
        One hot encoding
        to convert the "old" exported int data via OHE to binary matrix
        http://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
        """

        num_classes = np.max(data) + 1
        encoded_data = to_categorical(data, num_classes=num_classes)
        # encoded_data = encoded_data.reshape((data.shape[0], data.shape[1], num_classes))
        return encoded_data

    def one_hot_encode_string(y):

        """
        One hot encoding
        to convert the "old" exported int data via OHE to binary matrix
        http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
        """
        encoder = LabelEncoder()
        encoder.fit(y[0])
        print(encoder.classes_)
        print(encoder.transform(encoder.classes_))
        out = []
        for i in y:
            encoded_Y = encoder.transform(i)
            out.append(encoded_Y)

        # encoded_Y = encoder.transform(y)
        return to_categorical(out)

    if one_hot_encoding:
        global X_test, X_train, Y_test, Y_train, class_weight
        X_test = one_hot_encode_string(X_test_old)
        X_train = one_hot_encode_string(X_train_old)
    else:
        X_test = X_test_old
        X_train = X_train_old

    class_weighting = clw.compute_class_weight('balanced', np.unique(Y_train_old), Y_train_old)
    for i in range(len(class_weighting)):
        class_weight.update({i: class_weighting[i]})

    print(class_weight)
    Y_test = one_hot_encode_int(Y_test_old)
    Y_train = one_hot_encode_int(Y_train_old)


def use_old_data(one_hot_encoding=True):
    """
    to reuse the "old" exported data
    """

    Y_train_old = np.genfromtxt(directory + '/Y_train.csv', delimiter=',', dtype='str')
    Y_test_old = np.genfromtxt(directory + '/Y_test.csv', delimiter=',', dtype='str')
    X_train_old = np.genfromtxt(directory + '/X_train.csv', delimiter=',', dtype='int16')
    X_test_old = np.genfromtxt(directory + '/X_test.csv', delimiter=',', dtype='int16')

    def one_hot_encode_int(data):

        """
        One hot encoding
        to convert the "old" exported int data via OHE to binary matrix
        http://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
        """

        num_classes = np.max(data) + 1
        encoded_data = to_categorical(data, num_classes=num_classes)
        encoded_data = encoded_data.reshape((data.shape[0], data.shape[1], num_classes))
        return encoded_data

    def one_hot_encode_string(y):

        """
        One hot encoding
        to convert the "old" exported int data via OHE to binary matrix
        http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
        """
        encoder = LabelEncoder()
        encoder.fit(y)
        print(encoder.classes_)
        print(encoder.transform(encoder.classes_))
        encoded_Y = encoder.transform(y)
        return to_categorical(encoded_Y)

    if one_hot_encoding:
        global X_test, X_train, Y_test, Y_train
        X_test = one_hot_encode_int(X_test_old)
        X_train = one_hot_encode_int(X_train_old)
    else:
        X_test = X_test_old
        X_train = X_train_old

    Y_test = one_hot_encode_string(Y_test_old)
    Y_train = one_hot_encode_string(Y_train_old)


def use_data_nanocomb(one_hot_encoding=True, repeat=True, use_spacer=True, maxLen=None):
    """
    to use the nanocomb exported data
    """

    Y_train_old = pd.read_csv(directory + '/Y_train.csv', delimiter='\t', dtype='str', header=None)[1].as_matrix()
    Y_test_old = pd.read_csv(directory + '/Y_test.csv', delimiter='\t', dtype='str', header=None)[1].as_matrix()
    X_train_old = pd.read_csv(directory + '/X_train.csv', delimiter='\t', dtype='str', header=None)[1].as_matrix()
    X_test_old = pd.read_csv(directory + '/X_test.csv', delimiter='\t', dtype='str', header=None)[1].as_matrix()

    create_val = False

    try:
        Y_val_old = pd.read_csv(directory + '/Y_val.csv', delimiter='\t', dtype='str', header=None)[1].as_matrix()
        X_val_old = pd.read_csv(directory + '/X_val.csv', delimiter='\t', dtype='str', header=None)[1].as_matrix()
        print("loaded validation set from: " + directory + '/Y_val.csv')
    except:
        print("create validation set from train")
        create_val = True

    def one_hot_encode_int(data):

        """
        One hot encoding
        http://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
        """

        num_classes = np.max(data) + 1
        encoded_data = to_categorical(data, num_classes=num_classes)
        encoded_data = encoded_data.reshape((data.shape[0], data.shape[1], num_classes))
        return encoded_data

    def encode_string(maxLen=None, x=[], y=[]):

        """
        One hot encoding for classes
        to convert the "old" exported int data via OHE to binary matrix
        http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

        for dna ony to int values
        """

        def pad_n_repeat_sequences(sequences, maxlen=None, dtype='int32',
                                   padding='pre', truncating='pre', value=0.):
            """extended version of pad_sequences()"""
            if not hasattr(sequences, '__len__'):
                raise ValueError('`sequences` must be iterable.')
            lengths = []
            for x in sequences:
                if not hasattr(x, '__len__'):
                    raise ValueError('`sequences` must be a list of iterables. '
                                     'Found non-iterable: ' + str(x))
                lengths.append(len(x))
            num_samples = len(sequences)
            if maxlen is None:
                maxlen = np.max(lengths)

            # take the sample shape from the first non empty sequence
            # checking for consistency in the main loop below.
            sample_shape = tuple()
            for s in sequences:
                if len(s) > 0:
                    sample_shape = np.asarray(s).shape[1:]
                    break

            # make new array and fill with input seqs
            x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
            for idx, s in enumerate(sequences):
                if not len(s):
                    continue  # empty list/array was found
                if truncating == 'pre':
                    trunc = s[-maxlen:]
                elif truncating == 'post':
                    trunc = s[:maxlen]
                else:
                    raise ValueError('Truncating type "%s" not understood' % truncating)

                # check `trunc` has expected shape
                trunc = np.asarray(trunc, dtype=dtype)
                if trunc.shape[1:] != sample_shape:
                    raise ValueError(
                        'Shape of sample %s of sequence at position %s is different from expected shape %s' %
                        (trunc.shape[1:], idx, sample_shape))

                if padding == 'post':
                    x[idx, :len(trunc)] = trunc
                elif padding == 'pre':
                    x[idx, -len(trunc):] = trunc
                else:
                    raise ValueError('Padding type "%s" not understood' % padding)

                if repeat:
                    # repeat seq multiple times
                    repeat_seq = np.array([], dtype=dtype)
                    while len(repeat_seq) < maxLen:
                        if use_spacer:
                            spacer_length = random.randint(1, 50)
                            spacer = [value for i in range(spacer_length)]
                            repeat_seq = np.append(repeat_seq, spacer)
                            repeat_seq = np.append(repeat_seq, trunc)
                        else:
                            repeat_seq = np.append(repeat_seq, trunc)
                    x[idx, :] = repeat_seq[-maxLen:]

            return x

        encoder = LabelEncoder()

        if len(x) > 0:
            # x = [list(dnaSeq) for dnaSeq in x]
            # x = pad_sequences(x, maxlen=maxLen, dtype='str', padding='pre', truncating='pre', value="-")
            # a = "ATGCN-"
            a = "ATGCN"
            encoder.fit(list(a))
            print(encoder.classes_)
            print(encoder.transform(encoder.classes_))
            out = []
            for i in x:
                dnaSeq = re.sub(r"[^ACGTUacgtu]+", 'N', i)
                encoded_X = encoder.transform(list(dnaSeq))
                out.append(encoded_X)

            # encoded_Y = encoder.transform(y)
            # out = pad_sequences(out, maxlen=maxLen, dtype='int16', padding='pre', truncating='pre', value=0)#value=encoder.transform(encoder.classes_)[-1]+1)
            out = pad_n_repeat_sequences(out, maxlen=maxLen, dtype='int32', truncating='pre', value=0)

            # normalize
            # out = out / (len(a) - 1)

            # return to_categorical(out).reshape((out.shape[0], out.shape[1], len(a)))
            return to_categorical(out)
            # return out.reshape((out.shape[0], out.shape[1], 1))
        else:
            encoder.fit(y)
            print(encoder.classes_)
            print(encoder.transform(encoder.classes_))
            encoded_Y = encoder.transform(y)
            return to_categorical(encoded_Y)

    if one_hot_encoding:
        global X_test, X_train, X_val, Y_test, Y_train, Y_val
        if maxLen == None:
            length = []
            x_sets = [X_test_old, X_train_old]
            if create_val == False:
                x_sets.append(X_val_old)

            for X in x_sets:
                for i in X:
                    length.append(len(i))
            length.sort()
            # plt.hist(length,bins=500,range=(0,20000))
            # plt.show()
            maxLen = length[int(len(length) * 0.95)]
            print(maxLen)

        X_test = encode_string(maxLen, x=X_test_old)
        X_train = encode_string(maxLen, x=X_train_old)
        if create_val == False:
            X_val = encode_string(maxLen, x=X_val_old)
    else:
        X_test = X_test_old
        X_train = X_train_old
        if create_val == False:
            X_val = X_val_old

    Y_test = encode_string(y=Y_test_old)
    Y_train = encode_string(y=Y_train_old)
    if create_val:
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=SEED,
                                                          stratify=Y_train)
    else:
        Y_val = encode_string(y=Y_val_old)


def filter_train_data(species_to_keep=[1, 2]):
    """
    to define which classes should be learned
    :param species_to_keep: array with classes/labels which should be included in the train data e.g. [0,2]
    :return:
    """
    global X_train, Y_train
    Y_train_int = np.argmax(Y_train, axis=-1)
    arr = np.zeros(X_train.shape[0], dtype=int)
    for species in species_to_keep:
        arr[Y_train_int == int(species)] = 1
    X_train = X_train[arr == 1, :]
    Y_train = Y_train[arr == 1]


# define baseline model
def baseline_model(design=1, epochs=50, fit=True, decay=False):
    """
    make a model
    :param design: parameter for complexity of the NN, 1 == 2 layer LSTM, 2 == 3 layer LSTM
    :param epochs: number of epochs to train
    :param fit: True == trains the model and returns the trained model, False == returns untrained model
    :param decay: True == Lr will decrease over training time, False == same Lr whole training time
    :return: NN model
    """
    model = Sequential()

    # implementation: one of {0, 1, or 2}. If set to 0, the RNN will use an implementation that uses fewer, larger matrix products, thus running faster on CPU but consuming more memory.
    # model.add(Embedding(5, 16)) # try not to use OHE --> slow cause vector more complex than binary vector length 5
    # GRU 25% faster but needs more epochs -> equal to LSTM
    if design == 1:
        model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[-1]), return_sequences=True))
        model.add(LSTM(32))
    if design == 2:
        model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[-1]), return_sequences=True))
        model.add(LSTM(32, return_sequences=True))
        model.add(LSTM(32))
    # the proper way to use dropout with a recurrent network: the same dropout
    # mask (the same pattern of dropped units) should be applied at every timestep
    # Every recurrent layer in Keras has two dropout-related arguments:
    # dropout , a float specifying the dropout rate for input units of the layer, and
    # recurrent_dropout , specifying the dropout rate of the recurrent units.

    # reduce number of dimensions https://github.com/fchollet/keras/issues/6351
    # model.add(Flatten())
    model.add(Dense(Y_train.shape[-1], activation='softmax'))
    model.summary()
    if decay:
        myAdam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.04)
        model.compile(optimizer=myAdam, loss='categorical_crossentropy', metrics=['acc'])
    else:
        # myAdam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['acc'])
    # model.compile(optimizer='RMSProp', loss='mae', metrics=['acc'])

    if fit:
        # checkpoint
        # http://machinelearningmastery.com/check-point-deep-learning-models-keras/
        filepath = directory + "/weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        predictions = prediction_history()
        callbacks_list = [checkpoint, predictions]
        # callbacks_list = []

        """The choice of timesteps will influence both:

        The internal state accumulated during the forward pass.
        The gradient estimate used to update weights on the backward pass.

        Note that by default, the internal state of the network is reset after each batch"""

        # history = model.fit(X_test, Y_test, epochs=5, batch_size=X_train.shape[1], validation_split=0.2)

        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list,
                  validation_data=(X_test, Y_test))
        return model
    else:
        return model


def model_for_plot(design=1, sampleSize=1, nodes=32, suffix="", epochs=100, dropout=0, timesteps="default",
                   faster=False, path="/home/go96bix/Dropbox/Masterarbeit/ML", voting=False, tensorboard=False,
                   gpus=False, snapShotEnsemble=False, shuffleTraining=True, batch_norm=False):
    """
    method to train a model with specified properties, saves training behavior in /$path/"history"+suffix+".csv"
    :param design: parameter for complexity of the NN, 0 == 2 layer GRU, 1 == 2 layer LSTM, 2 == 3 layer LSTM
    :param sampleSize: fraction of samples that will be used for training (1/samplesize). 1 == all samples, 2 == half of the samples
    :param nodes: number of nodes per layer
    :param suffix: suffix for output files
    :param epochs: number of epochs to train
    :param dropout: rate of dropout to use, 0 == no Dropout, 0.2 = 20% Dropout
    :param timesteps: size of "memory" of LSTM, don't change if not sure what you're doing
    :param faster: speedup due higher batch size, can reduce accuracy
    :param path: define the directory where the training history should be saved
    :param voting: if true than saves the history of the voting / mean-predict subsequences, reduces training speed
    :param tensorboard: for observing live changes to the network, more details see web
    :param cuda: use GPU for calc, not tested jet, not working
    :return: dict with loss and model
    """
    model = Sequential()
    global batch_size, X_train
    if timesteps == "default":
        timesteps = X_train.shape[1]
    if faster:
        batch = batch_size * 16
    else:
        batch = batch_size
    # if gpus:
    #     LSTM = keras.layers.CuDNNLSTM
    if design == 0:
        model.add(GRU(nodes, input_shape=(timesteps, X_train.shape[-1]), return_sequences=True, dropout=dropout))
        model.add(GRU(nodes, dropout=dropout))

    if design == 1:
        model.add(LSTM(nodes, input_shape=(timesteps, X_train.shape[-1]), return_sequences=True, dropout=dropout))
        model.add(LSTM(nodes, dropout=dropout))

    if design == 2:
        model.add(LSTM(nodes, input_shape=(timesteps, X_train.shape[-1]), return_sequences=True, dropout=dropout))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(LSTM(nodes, return_sequences=True, dropout=dropout))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(LSTM(nodes, dropout=dropout))
        if batch_norm:
            model.add(BatchNormalization())

    if design == 3:
        model.add(LSTM(nodes, input_shape=(timesteps, X_train.shape[-1]), return_sequences=True, dropout=dropout))
        model.add(LSTM(nodes, return_sequences=True, dropout=dropout))
        model.add(LSTM(nodes, return_sequences=True, dropout=dropout))
        model.add(LSTM(nodes, dropout=dropout))

    if design == 4:
        model.add(Bidirectional(LSTM(nodes, return_sequences=True, dropout=dropout),
                                input_shape=(timesteps, X_train.shape[-1])))
        model.add(Bidirectional(LSTM(nodes, return_sequences=True, dropout=dropout)))
        model.add(Bidirectional(LSTM(nodes, dropout=dropout)))

    if design == 5:
        model.add(Conv1D(nodes, 9, input_shape=(timesteps, X_train.shape[-1]), activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(nodes, 9, activation='relu'))
        # model.add(layers.MaxPooling1D(3))
        model.add(Bidirectional(
            LSTM(nodes, return_sequences=True, dropout=dropout)))
        model.add(Bidirectional(LSTM(nodes, dropout=dropout)))

    if design == 6:
        # This returns a tensor
        inputs = Input(shape=(timesteps, X_train.shape[-1]))

        left1 = Bidirectional(LSTM(nodes, return_sequences=True, dropout=dropout))(inputs)
        left2 = Bidirectional(LSTM(nodes, dropout=dropout))(left1)

        right = Conv1D(nodes, 9, activation='relu')(inputs)
        right = MaxPooling1D(3)(right)
        right = Conv1D(nodes, 9, activation='relu')(right)
        right = MaxPooling1D(3)(right)
        right3 = Conv1D(nodes, 9, activation='relu')(right)
        right_flat = Flatten()(right3)

        joined = Concatenate()([left2, right_flat])
        predictions = Dense(Y_train.shape[-1], activation='softmax')(joined)

        model = Model(inputs=inputs, outputs=predictions)

    if design == 7:
        model.add(Conv1D(nodes, 9, input_shape=(timesteps, X_train.shape[-1]), activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(nodes, 9, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(nodes, 9, activation='relu'))
        # model.add(layers.MaxPooling1D(3))
        model.add(Bidirectional(
            LSTM(nodes, return_sequences=True, dropout=dropout, recurrent_dropout=0.5)))
        model.add(Bidirectional(LSTM(nodes, dropout=dropout, recurrent_dropout=0.5)))

    if design == 8:
        model.add(Conv1D(nodes, 9, input_shape=(timesteps, X_train.shape[-1]), activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(nodes, 9, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(nodes, 9, activation='relu'))
        # model.add(layers.MaxPooling1D(3))
        model.add(Bidirectional(
            LSTM(nodes, return_sequences=True, dropout=dropout, recurrent_dropout=0.2)))
        model.add(Bidirectional(LSTM(nodes, return_sequences=True, dropout=dropout, recurrent_dropout=0.2)))
        model.add(Bidirectional(LSTM(nodes, dropout=dropout, recurrent_dropout=0.2)))

    if design != 6:
        model.add(Dense(Y_train.shape[-1], activation='softmax'))

    model.summary()
    # adam = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
    if gpus >= 2:
        model_basic = model
        with tf.device("/cpu:0"):
            # initialize the model
            # model = MiniGoogLeNet.build(width=32, height=32, depth=3,
            #                             classes=10)
            model = model_basic

        # make the model parallel
        parallel_model = multi_gpu_model(model, gpus=gpus)
        model = parallel_model

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    filepath = directory + "/weights.best." + suffix + ".hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    predictions = prediction_history()
    time_callback = TimeHistory()

    callbacks_list = [checkpoint, predictions, time_callback]
    if voting:
        myAccuracy = accuracyHistory()
        callbacks_list.append(myAccuracy)
    if snapShotEnsemble:
        nb_snapshots = 5
        snapshot = SnapshotCallbackBuilder(epochs, nb_snapshots, 0.1)
        lr_manipulation = lrManipulator(nb_epochs=epochs, nb_snapshots=nb_snapshots)
        callbacks_list.extend(snapshot.get_callbacks(model_prefix=suffix + "Model", fn_prefix=path + "/weights"))
        callbacks_list.append(lr_manipulation)

    if tensorboard:
        if not os.path.isdir(path + '/my_log_dir'):
            os.makedirs(path + '/my_log_dir')
        tensorboard = keras.callbacks.TensorBoard(
            # Log files will be written at this location
            log_dir=path + '/my_log_dir',
            # We will record activation histograms every 1 epoch
            histogram_freq=1,
            # We will record embedding data every 1 epoch
            embeddings_freq=1,
        )
        tensorboard = keras.callbacks.TensorBoard(log_dir=path + '/my_log_dir', histogram_freq=0, batch_size=32,
                                                  write_graph=True, write_grads=False, write_images=False,
                                                  embeddings_freq=0, embeddings_layer_names=None,
                                                  embeddings_metadata=None)
        callbacks_list.append(tensorboard)

    # class_weight = {0: 1.,
    #                 1: 11.,
    #                 }
    # class_weight = clw.compute_class_weight('balanced', np.unique(Y_train), Y_train)

    hist = model.fit(X_train[0:int(len(X_train) / sampleSize)], Y_train[0:int(len(X_train) / sampleSize)],
                     epochs=epochs, batch_size=batch, callbacks=callbacks_list,
                     validation_data=(X_val, Y_val), class_weight=class_weight, shuffle=shuffleTraining)
    times = time_callback.times
    if voting:
        acc_votes = myAccuracy.normalVote_train
        acc_means = myAccuracy.meanVote_train
        val_acc_votes = myAccuracy.normalVote_val
        val_acc_means = myAccuracy.meanVote_val

    if not os.path.isfile(path + "/history" + suffix + ".csv"):
        histDataframe = pd.DataFrame(hist.history)
        histDataframe = histDataframe.assign(time=times)
        if voting:
            histDataframe = histDataframe.assign(acc_vote=acc_votes)
            histDataframe = histDataframe.assign(acc_mean=acc_means)
            histDataframe = histDataframe.assign(val_acc_vote=val_acc_votes)
            histDataframe = histDataframe.assign(val_acc_mean=val_acc_means)
        histDataframe.to_csv(path + "/history" + suffix + ".csv")
    else:
        histDataframe = pd.DataFrame(hist.history)
        histDataframe = histDataframe.assign(time=times)
        if voting:
            histDataframe = histDataframe.assign(acc_vote=acc_votes)
            histDataframe = histDataframe.assign(acc_mean=acc_means)
            histDataframe = histDataframe.assign(val_acc_vote=val_acc_votes)
            histDataframe = histDataframe.assign(val_acc_mean=val_acc_means)
        histDataframe.to_csv(path + "/history" + suffix + ".csv", mode='a', header=False)
    # model.save(directory + "/deep-model-longrun" + ".h5")
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'model': model}


def calc_predictions(X, Y, y_pred, do_print=False):
    """
    plot predictions
    :param X: raw-data which should be predicted
    :param Y: true labels for X
    :param do_print: True == print the cross-tab of the prediction
    :param y_pred: array with predicted labels for X
    :return: y_true_small == True labels for complete sequences, yTrue == True labels for complete subsequences, y_pred_mean == with mean predicted labels for complete sequences, y_pred_voted == voted labels for complete sequences, y_pred == predicted labels for complete subsequences
    """

    def print_predictions(y_true, y_pred, y_true_small, y_pred_voted, y_pred_sum, y_pred_mean_weight_std,
                          y_pred_mean_weight_ent):
        table = pd.crosstab(
            pd.Series(y_true),
            pd.Series(y_pred),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print("standard version")
        print(table)
        accuracy = metrics.accuracy_score(y_true, y_pred) * 100
        print("standard version")
        print("acc = " + str(accuracy))

        table = pd.crosstab(
            pd.Series(y_true_small),
            pd.Series(y_pred_voted),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print("vote version")
        print(table)
        accuracy = metrics.accuracy_score(y_true_small, y_pred_voted) * 100
        print("vote version")
        print("acc = " + str(accuracy))

        table = pd.crosstab(
            pd.Series(y_true_small),
            pd.Series(y_pred_sum),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print("mean version")
        print(table)
        accuracy = metrics.accuracy_score(y_true_small, y_pred_sum) * 100
        print("mean version")
        print("acc = " + str(accuracy))

        table = pd.crosstab(
            pd.Series(y_true_small),
            pd.Series(y_pred_mean_weight_ent),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print("entropie version")
        print(table)
        accuracy = metrics.accuracy_score(y_true_small, y_pred_mean_weight_ent) * 100
        print("entropie version")
        print("acc = " + str(accuracy))

        table = pd.crosstab(
            pd.Series(y_true_small),
            pd.Series(y_pred_sum),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print("std version")
        print(table)
        accuracy = metrics.accuracy_score(y_true_small, y_pred_mean_weight_std) * 100
        print("std version")
        print("acc = " + str(accuracy))

    # for addition of probability and not voting
    y_pred_mean = []
    y_pred_mean_exact = []
    weigth_entropy = []
    y_pred_mean_weight_ent = []
    weigth_std = []
    y_pred_mean_weight_std = []

    for i in y_pred:
        # standard distribution of values
        weigth_std.append(np.std(i))
        # weigth_std.append(np.std(i)**5)
        # weigth_std.append(np.var(i))

        # entropie if this values corresbond to a normal distribution
        weigth_entropy.append(scipy.stats.entropy(scipy.stats.norm.pdf(i, loc=0.5, scale=0.25)))

    for i in range(0, int(len(y_pred) / batch_size)):
        sample_pred_mean = np.array(np.sum(y_pred[i * batch_size:i * batch_size + batch_size], axis=0) / batch_size)
        y_pred_mean.append(np.argmax(sample_pred_mean))
        y_pred_mean_exact.append(sample_pred_mean)

        sample_weigths = weigth_entropy[i * batch_size:i * batch_size + batch_size]
        sw_normalized = np.array(sample_weigths / np.sum(sample_weigths)).reshape(-1, 1)
        # add entropy weighted prediction
        y_pred_mean_weight_ent.append(np.argmax(np.array(
            np.sum(np.array(y_pred[i * batch_size:i * batch_size + batch_size]) * sw_normalized, axis=0) / batch_size)))

        sample_weigths = weigth_std[i * batch_size:i * batch_size + batch_size]
        sw_normalized = np.array(sample_weigths / np.sum(sample_weigths)).reshape(-1, 1)
        # add standard deviation weighted prediction
        y_pred_mean_weight_std.append(np.argmax(np.array(
            np.sum(np.array(y_pred[i * batch_size:i * batch_size + batch_size]) * sw_normalized, axis=0) / batch_size)))

        # print(y_pred[i * batch_size:i * batch_size + batch_size])
    # standard respond
    y_pred = np.argmax(y_pred, axis=-1)

    # count all votes for one big sequence
    y_true = np.argmax(Y, axis=-1)
    y_true_small, y_pred_voted = [], []
    """calc voting of sequence (via votings for subsequences)"""
    for i in range(0, int(len(y_true) / batch_size)):
        arr = np.array(np.bincount(y_pred[i * batch_size:i * batch_size + batch_size]))
        best = np.argwhere(arr == np.amax(arr)).flatten()
        y_pred_voted.append(np.random.permutation(best)[0])

        # y_pred_voted.append(np.argmax(np.array(np.bincount(y_pred[i * batch_size:i*batch_size+batch_size]))))
        # print(np.bincount(y_true[i * batch_size:i * batch_size + batch_size]))
        y_true_small.append(np.argmax(np.array(np.bincount(y_true[i * batch_size:i * batch_size + batch_size]))))

    if do_print:
        print_predictions(y_true, y_pred, y_true_small, y_pred_voted, y_pred_mean, y_pred_mean_weight_std,
                          y_pred_mean_weight_ent)
    return y_true_small, y_pred_mean, y_pred_voted, y_pred, np.array(y_pred_mean_exact)


def plot_histogram(pred, true, labels):
    """
    plots the distribution of the softmax values / predictions for the classes
    Input: predicted likelihood of classes and real classes
    Output: histogram with distributions over multiple classes
    """
    colors = ["steelblue", "saddlebrown", "seagreen", "mediumorchid"]
    fig = plt.figure()
    if (len(true.shape) > 1):
        y_true = np.argmax(true, axis=-1)
    else:
        y_true = true
    for species in range(pred.shape[1]):
        ax = fig.add_subplot(3, 1, (species + 1))
        votes = pred[y_true == species]
        ax.hist(votes[:, species], 50, range=(0, 1), log=True, facecolor=colors[species], alpha=0.75)
        # plt.yscale('log')
        plt.xlim(0, 1)
        # plt.ylim(0,5000)
        ax.set_title('Prediction Distribution for ' + labels[species])
        plt.xlabel('Predicted Likelihood')
        plt.ylabel('# Predictions')
    # plt.legend()

    plt.show()


def snap_Shot_ensemble(M=5, nb_epoch=100, alpha_zero=0.1, model_prefix='Model_',
                       path="/home/go96bix/Dropbox/Masterarbeit/ML"):
    """
    first try to use snapshots
    :param M: number of snapshots
    :param nb_epoch: T = number of epochs
    :param alpha_zero: initial learning rate
    :param model_prefix: output name
    :param path: define the directory where the training history should be saved
    :return:
    """

    snapshot = SnapshotCallbackBuilder(nb_epoch, M, alpha_zero)
    model = baseline_model(fit=False, decay=False)

    # filepath = directory+"/weights.best."+suffix+".hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    predictions = prediction_history()
    time_callback = TimeHistory()
    myAccuracy = accuracyHistory()
    lr_manipulation = lrManipulator(nb_epochs=nb_epoch, nb_snapshots=M)
    callbacks_list = snapshot.get_callbacks(model_prefix=model_prefix)
    callbacks_list.append(predictions)
    callbacks_list.append(time_callback)
    callbacks_list.append(lr_manipulation)
    callbacks_list.append(myAccuracy)

    hist = model.fit(X_train, Y_train, epochs=nb_epoch, batch_size=batch_size,
                     callbacks=callbacks_list,
                     validation_data=(X_test, Y_test))
    times = time_callback.times
    acc_votes = myAccuracy.normalVote_train
    acc_means = myAccuracy.meanVote_train
    val_acc_votes = myAccuracy.normalVote_val
    val_acc_means = myAccuracy.meanVote_val

    if not os.path.isfile(path + "/history" + model_prefix + ".csv"):
        histDataframe = pd.DataFrame(hist.history)
        histDataframe = histDataframe.assign(time=times)
        histDataframe = histDataframe.assign(acc_vote=acc_votes)
        histDataframe = histDataframe.assign(acc_mean=acc_means)
        histDataframe = histDataframe.assign(val_acc_vote=val_acc_votes)
        histDataframe = histDataframe.assign(val_acc_mean=val_acc_means)
        histDataframe.to_csv(path + "/history" + model_prefix + ".csv")
    else:
        histDataframe = pd.DataFrame(hist.history)
        histDataframe = histDataframe.assign(time=times)
        histDataframe = histDataframe.assign(acc_vote=acc_votes)
        histDataframe = histDataframe.assign(acc_mean=acc_means)
        histDataframe = histDataframe.assign(val_acc_vote=val_acc_votes)
        histDataframe = histDataframe.assign(val_acc_mean=val_acc_means)
        histDataframe.to_csv(path + "/history" + model_prefix + ".csv", mode='a',
                             header=False)


def prediction_from_Ensemble(nb_classes, X, Y, dir="/home/go96bix/weights", calc_weight=False, weights=[], mean=True,
                             multiBatch=False):
    """
    loads models and returns prediction weights or prints the accuracy reached with predefined weights
    :param nb_classes: how many different classes/labels exist
    :param X: raw-data which should be predicted
    :param Y: true labels for X
    :param dir: define the directory where the models are saved
    :param calc_weight: True == calculates best weighting of the models
    :param weights: array with pre defined weights for models, only set if you do not want to calc optimal weights
    :return: returns weights
    """

    def weighted_ensemble(preds, nb_classes, nb_models, X, Y, NUM_TESTS=250):
        """
        calculates the best weights
        :param preds: array with predicted labels for X
        :param nb_classes: how many different classes/labels exist
        :param nb_models: how many different models exist
        :param X: raw-data which should be predicted
        :param Y: true labels for X
        :param NUM_TESTS: how many test should be done for the derteming the best weight
        :return: array with best weights
        """

        # Create the loss metric
        def log_loss_func(weights, X, Y, preds, nb_classes):
            ''' scipy minimize will pass the weights as a numpy array
            https://github.com/titu1994/Snapshot-Ensembles/blob/master/optimize_cifar100.ipynb
            '''
            if multiBatch:
                final_prediction = np.zeros((X.shape[1], nb_classes), dtype='float32')
            else:
                final_prediction = np.zeros((X.shape[0], nb_classes), dtype='float32')

            for weight, prediction in zip(weights, preds):
                final_prediction += weight * prediction

            return log_loss(np.argmax(Y, axis=-1), final_prediction)

        best_acc = 0.0
        best_weights = None

        # Parameters for optimization
        constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
        bounds = [(0, 1)] * len(preds)
        foo = []
        # Check for NUM_TESTS times
        for iteration in range(NUM_TESTS):
            # Random initialization of weights
            prediction_weights = np.random.random(nb_models)

            # Minimise the loss
            result = minimize(log_loss_func, prediction_weights, args=(X, Y, preds, nb_classes), method='SLSQP',
                              bounds=bounds, constraints=constraints)
            print('Best Ensemble Weights: {weights}'.format(weights=result['x']))
            weights = result['x']
            foo.append(weights)
            y_true_small, y_true, y_pred_mean, y_pred_voted, y_pred = calculate_weighted_accuracy(weights, preds,
                                                                                                  nb_classes, X=X, Y=Y,
                                                                                                  multiBatch=multiBatch)

            if mean:
                accuracy = metrics.accuracy_score(y_true_small, y_pred_mean) * 100
                # accuracy = metrics.accuracy_score(yTrue, y_pred) * 100
                print("accuracy with mean: " + str(accuracy))
                print("-----------------------------------------")
                # Save current best weights
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_weights = weights
            else:
                accuracy = metrics.accuracy_score(y_true, y_pred) * 100
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_weights = weights

        print("Best accuracy" + str(best_acc))
        print("Best weigths" + str(best_weights))
        return best_weights

    models_filenames = []

    if os.path.isfile(dir):
        models_filenames.append(dir)
    else:
        for file in sorted(os.listdir(dir)):
            if not file.endswith("Best.h5") and file.endswith(".h5"):
                print(file)
                models_filenames.append(os.path.join(dir, file))

    if len(models_filenames) > 1:
        assert multiBatch != True, f"if multibatch == True, only use one model, you try to use: {models_filenames}"
    if len(models_filenames) == 1:
        assert multiBatch == True, f"if multibatch == False, use more than one model, you try to use: {models_filenames}"

    use_data_nanocomb(one_hot_encoding=True)
    # use_old_data(one_hot_encoding=True)
    shrink_timesteps()
    # batch_size = 100

    preds = []
    yPreds = []

    for fn in models_filenames:
        print("load model and predict")
        model = load_model(fn)
        # model.load_weights(fn)
        yPreds = model.predict(X, batch_size=batch_size)
        preds.append(yPreds)

    y_true_small, y_pred_mean, y_pred_voted, y_pred, y_pred_mean_exact = calc_predictions(X, Y, do_print=True,
                                                                                          y_pred=yPreds)

    if multiBatch:
        models_filenames = []
        X_reorder = []
        single_SubS = []
        Y_reorder = []
        preds = []
        for i in range(batch_size):
            for k in range(yPreds.shape[0] // batch_size):
                preds.append(yPreds[k * batch_size + i])
                single_SubS.append(X[k * batch_size + i])
                if i == 0:
                    Y_reorder.append(Y[k * batch_size + i])
            X_reorder.append(single_SubS)
            single_SubS = []
            models_filenames.append(f"SubSam {i}")

        X = np.array(X_reorder)
        Y = Y_reorder
        preds = np.array(preds).reshape((batch_size, (np.array(preds).size // nb_classes) // batch_size, nb_classes))

    prediction_weights = [1. / len(models_filenames)] * len(models_filenames)

    calculate_weighted_accuracy(prediction_weights, preds, nb_classes, X=X, Y=Y, multiBatch=multiBatch)

    if calc_weight == True:
        best_weights = weighted_ensemble(preds, nb_classes, nb_models=len(models_filenames), X=X, Y=Y)
        return best_weights
    elif len(weights) > 0:
        calculate_weighted_accuracy(weights, preds, nb_classes, X=X, Y=Y)
    else:
        return prediction_weights


def calculate_weighted_accuracy(prediction_weights, preds, nb_classes, X, Y, multiBatch=True):
    """
    equally weighted model prediction accuracy
    :param prediction_weights: array with weights of single models e.g. [0,0.6,0.4]
    :param preds: array with the predicted classes/labels of the models
    :param nb_classes: how many different classes/labels exist
    :param X: raw-data which should be predicted
    :param Y: true labels for X
    :return: y_true_small == True labels for complete sequences, yTrue == True labels for complete subsequences, y_pred_mean == with mean predicted labels for complete sequences, y_pred_voted == voted labels for complete sequences, y_pred == predicted labels for complete subsequences
    """
    if multiBatch:
        weighted_predictions = np.zeros((X.shape[1], nb_classes), dtype='float32')
    else:
        weighted_predictions = np.zeros((X.shape[0], nb_classes), dtype='float32')

    for weight, prediction in zip(prediction_weights, preds):
        weighted_predictions += weight * prediction

    yPred = np.argmax(weighted_predictions, axis=1)
    yTrue = np.argmax(Y, axis=-1)
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy ensemble: ", accuracy)
    print("Error ensemble: ", error)
    y_true_small, y_pred_mean, y_pred_voted, y_pred, y_pred_mean_exact = calc_predictions(X=X, Y=Y,
                                                                                          y_pred=weighted_predictions)
    # added yTrue to output
    return y_true_small, yTrue, y_pred_mean, y_pred_voted, y_pred


def run_tests_for_plotting():
    """
    setups for some of the tests from the masterthesis
    :return:
    """
    # test_and_plot(path='/home/go96bix/projects/nanocomb/nanocomb/plots', suffix="baseline_design2",
    #               do_shrink_timesteps=False, design=2, nodes=100, faster=True, titel="Accuracy no repeat",
    #               accuracy=True, epochs=50, repeat=False, batch_norm=False)
    #
    # test_and_plot(path='/home/go96bix/projects/nanocomb/nanocomb/plots', suffix="design1_spacer-repeat",
    #               do_shrink_timesteps=False, design=1, nodes=100, faster=True, titel="Accuracy with spacer-repeat",
    #               accuracy=True, epochs=50, repeat=True, use_repeat_spacer=True, batch_norm=False)
    #
    # test_and_plot(path='/home/go96bix/projects/nanocomb/nanocomb/plots', suffix="design1_normal-repeat",
    #               do_shrink_timesteps=False, design=1, nodes=100, faster=True, titel="Accuracy with normal-repeat",
    #               accuracy=True, epochs=50, repeat=True, use_repeat_spacer=False, batch_norm=False)
    #
    # test_and_plot(path='/home/go96bix/projects/nanocomb/nanocomb/plots', suffix="design1_normal-repeat-TBTT",
    #               do_shrink_timesteps=True, design=1, nodes=100, faster=False, titel="Accuracy with normal-repeat and TBTT",
    #               accuracy=True, epochs=50, repeat=True, use_repeat_spacer=False, batch_norm=False)
    #
    # test_and_plot(path='/home/go96bix/projects/nanocomb/nanocomb/plots', suffix="design4_baseline",
    #               do_shrink_timesteps=False, design=4, nodes=100, faster=True,
    #               titel="Accuracy BiDir LSTM",
    #               accuracy=True, epochs=50, repeat=False, use_repeat_spacer=False, batch_norm=False)
    #
    test_and_plot(path='/home/go96bix/projects/nanocomb/nanocomb/plots', suffix="design4_emanuel",
                  do_shrink_timesteps=True, voting=False, design=4, nodes=100, faster=True,
                  titel="Accuracy BiDir LSTM normal-repeat and TBTT",
                  accuracy=True, epochs=15, repeat=True, use_repeat_spacer=False, batch_norm=False)
    #
    # test_and_plot(path='/home/go96bix/projects/nanocomb/nanocomb/plots', suffix="design7",
    #               do_shrink_timesteps=False, design=7, nodes=100, faster=True,
    #               titel="Accuracy 3 layer Conv1D & LSTM",
    #               accuracy=True, epochs=50, repeat=False, use_repeat_spacer=False, batch_norm=False,dropout=0.1)

    # test_and_plot(path='/home/go96bix/projects/nanocomb/nanocomb/plots', suffix="design8_long-run",
    #               do_shrink_timesteps=False, design=8, nodes=100, faster=True,
    #               titel="Accuracy 3 layer Conv1D & 3 layer LSTM",
    #               accuracy=True, epochs=150, repeat=False, use_repeat_spacer=False, batch_norm=False,dropout=0.1)
    # test_and_plot(path='/home/go96bix/projects/nanocomb/nanocomb/plots', suffix="design6_short_training",
    #               do_shrink_timesteps=False, design=6, nodes=100, faster=True,
    #               titel="Accuracy left LSTM und Conv1D",accuracy=True, epochs=10, repeat=False, use_repeat_spacer=False,
    #               batch_norm=False)


def test_and_plot(path, suffix, batch_norm=False, filter_trainset=False, use_old_dataset=False,
                  do_shrink_timesteps=True,
                  one_hot_encoding=True, repeat=True, use_repeat_spacer=False, val_size=0.3, input_subSeqlength=0,
                  design=1, sampleSize=1, nodes=32,
                  snapShotEnsemble=False, epochs=100, dropout=0, timesteps="default", faster=False,
                  voting=False, tensorboard=False, gpus=False, titel='', x_axes='', y_axes='', accuracy=False,
                  loss=False, runtime=False, label1='', label2='', label3='', label4=''):
    """
    1. gets settings and prepare data
    2. saves settings
    3. starts training
    4. saves history
    5. plots results
    :return:
    """
    # GET SETTINGS AND PREPARE DATA
    global X_train, X_test, X_val, Y_train, Y_test, Y_val, batch_size, SEED, directory
    if use_old_dataset:
        use_old_data(one_hot_encoding=one_hot_encoding)
    else:
        use_data_nanocomb(one_hot_encoding=one_hot_encoding, repeat=repeat, use_spacer=use_repeat_spacer)

    if len(X_val) == 0:
        print("make new val set")
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_size, random_state=SEED,
                                                          stratify=Y_train)
    else:
        print("val set already exits")

    if do_shrink_timesteps:
        if input_subSeqlength:
            shrink_timesteps(input_subSeqlength)
        else:
            shrink_timesteps()

    """to limit the training on specified classes/hosts"""
    if filter_trainset:
        filter_train_data()

    # SAVE SETTINGS
    with open(path + '/' + suffix + "_config.txt", "w") as file:
        for i in locals().items():
            file.write(str(i) + '\n')
        if faster:
            file.write('(\'batchsize\', ' + str(batch_size * 10) + ')\n')
        else:
            file.write('(\'batchsize\', ' + str(batch_size) + ')\n')
        file.write('(\'SEED\', ' + str(SEED) + ')\n')
        file.write('(\'directory\', ' + str(directory) + ')\n')

    # START TRAINING

    model_for_plot(design=design, sampleSize=sampleSize, nodes=nodes, suffix=suffix, epochs=epochs, dropout=dropout,
                   timesteps=timesteps, faster=faster, path=path, voting=voting, tensorboard=tensorboard, gpus=gpus,
                   snapShotEnsemble=snapShotEnsemble, shuffleTraining=not do_shrink_timesteps, batch_norm=batch_norm)

    plotting_results.plotting_history(path=path, file="history" + suffix + ".csv", titel=titel, x_axes=x_axes,
                                      y_axes=y_axes,
                                      accuracy=accuracy, loss=loss, voting=voting, runtime=runtime, label1=label1,
                                      label2=label2, label3=label3, label4=label4)


if __name__ == '__main__':
    """settings"""
    X_test = []
    X_val = []
    X_train = []
    Y_test = []
    Y_val = []
    Y_train = []
    class_weight = {}
    SEED = 42
    new_model = False
    filter_trainset = False
    use_old_dataset = False
    do_shrink_timesteps = True
    batch_size = 5  # X_train.shape[0]

    """define the folder where to find the training/testing data"""
    suffix = '100Samples'

    directory = '/home/go96bix/projects/nanocomb/nanocomb/' + suffix

    if use_old_dataset:
        use_old_data(one_hot_encoding=True)
    else:
        use_data_nanocomb()
    #

    # X_test = X_test[0:10]
    # Y_test = Y_test[0:10]

    if do_shrink_timesteps:
        shrink_timesteps()  # input_subSeqlength=1000)

    """to limit the training on specified classes/hosts"""
    if filter_trainset:
        filter_train_data()

    """if interested in cross validation"""
    # classic
    # estimator = KerasClassifier(build_fn=baseline_model(fit=False), epochs=50,  batch_size=batch_size,verbose=1)
    #
    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    # results = cross_val_score(estimator, X_train, Y_train, cv=kfold)
    # print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    # tweeked
    # https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
    # https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
    # estimator = KerasClassifier(build_fn=baseline_model(fit=False), epochs=5,  batch_size=batch_size*100,verbose=1)
    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    # for train, val in kfold.split(X_train, Y_test):
    #     weights = [] # hier ueberlegen wie ich random weights init und dann alte wiederverwendet als init fuer naechste model
    #     model = baseline_model(epochs=5,fit=False)
    #     model.fit(model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list,validation_data=(X_test, Y_test)))
    # weights = prediction_from_Ensemble(51,X_val,Y_val,dir="/home/go96bix/projects/nanocomb/nanocomb/plots/weights/complex/",calc_weight=True, mean=False)
    weights = prediction_from_Ensemble(51, X_val, Y_val,
                                       dir="/home/go96bix/projects/nanocomb/nanocomb/weights.best.design4_repeat_vote.hdf5",
                                       calc_weight=True, mean=False, multiBatch=True)
    prediction_from_Ensemble(51, X_test, Y_test, weights=weights,
                             dir="/home/go96bix/projects/nanocomb/nanocomb/weights.best.design4_repeat_vote.hdf5", multiBatch=True)
    # # print(weights)

    # run_tests_for_plotting()
    exit()
    # test_and_plot(path='/home/go96bix/projects/nanocomb/nanocomb/plots', suffix="long_bigNN_test", snapShotEnsemble=False,
    #               do_shrink_timesteps=False, design=3, nodes=100, faster=True, titel="Accuracy (n-1 Ebola-set)",
    #               accuracy=True, epochs=150,repeat=False)#, dropout=0.1)
    # exit()
    # model_for_plot(voting=False, suffix="nanocomb_test_100samples_len10000_noVote_Design2", epochs=100, faster=True,
    #                nodes=64, path='/home/go96bix/projects/nanocomb/nanocomb/plots', tensorboard=False, design=2)
    # # # # snap_Shot_ensemble(nb_epoch=100,M=5,model_prefix="Flavi_m                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           yVersion")
    # # #
    # exit()
    model = load_model(
        "/home/go96bix/projects/nanocomb/nanocomb/weights.best.design4_repeat_vote.hdf5")
    # "/home/go96bix/projects/nanocomb/nanocomb/models/weights.best.design1_normal-repeat-TBTT.hdf5")
    pred = model.predict(X_test)
    # np.savetxt('/home/go96bix/projects/nanocomb/nanocomb/pred_vector_train.csv', pred, fmt='%s', delimiter=',')
    # exit()
    # score, acc = model.evaluate(X_test, Y_test, verbose=0)
    # print('Test accuracy:', acc)
    y_true_small, y_pred_mean, y_pred_voted, y_pred, y_pred_mean_exact = calc_predictions(X_test, Y_test, y_pred=pred,
                                                                                          do_print=True)
    # pred = np.argmax(pred, axis=-1)
    print(Y_test)
    # print(pred)
    exit()
    species = ['Homo sapiens', 'Macaca', 'Rousettus']
    plot_histogram(pred, Y_train, species)
    plot_histogram(y_pred_mean_exact, np.array(y_true_small), species)

    # calc_predictions(X_test,Y_test,y_pred=pred)

"""
good idea to increase the capacity of your network until overfitting becomes your primary obstacle
As long as you are not overfitting too badly, then you are likely under-capacity. Site 227
"""
