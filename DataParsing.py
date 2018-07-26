import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
import random
from keras.utils import to_categorical
from logging import warning
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing

def encode_string(maxLen=None, x=[], y=[], y_encoder=None, repeat=True, use_spacer=True, online_Xtrain_set=False):
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
        a = "ATGCN-"

        encoder.fit(list(a))
        # print(encoder.classes_)
        # print(encoder.transform(encoder.classes_))
        out = []
        for i in x:
            dnaSeq = re.sub(r"[^ACGTUacgtu]+", 'N', i)
            encoded_X = encoder.transform(list(dnaSeq))
            out.append(encoded_X)

        if online_Xtrain_set:
            X_train_categorial = []
            for seq in out:
                X_train_categorial.append(np.array(to_categorical(seq, num_classes=len(a)), dtype=np.bool))
            return X_train_categorial
        else:
            out = pad_n_repeat_sequences(out, maxlen=maxLen, dtype='int16', truncating='pre', value=0)

        return np.array(to_categorical(out, num_classes=len(a)), dtype=np.bool)
    else:
        if y_encoder != None:
            encoder.fit(y)
            # print(encoder.classes_)
            # print(encoder.transform(encoder.classes_))
            if np.array(encoder.classes_ != y_encoder.classes_).all():
                warning(f"Warning not same classes in training and test set")
            useable_classes = set(encoder.classes_).intersection(y_encoder.classes_)
            # print(useable_classes)
            try:
                assert np.array(encoder.classes_ == y_encoder.classes_).all()
            except AssertionError:
                warning(
                    f"not all test classes in training data, only {useable_classes} predictable "
                    f"from {len(encoder.classes_)} different classes\ntest set will be filtered so only predictable"
                    f" classes are included")

            try:
                assert len(useable_classes) == len(encoder.classes_)
            except AssertionError:
                print(f"not all test classes in training data, only " \
                      f"{useable_classes} predictable from {len(encoder.classes_)} different classes" \
                      f"\ntest set will be filtered so only predictable classes are included")

            if not len(useable_classes) == len(encoder.classes_):
                global X_test, Y_test
                arr = np.zeros(X_test.shape[0], dtype=int)
                for i in useable_classes:
                    arr[y == i] = 1

                X_test = X_test[arr == 1, :]
                y = y[arr == 1]
                encoded_Y = y_encoder.transform(y)
            else:
                encoded_Y = encoder.transform(y)

            return to_categorical(encoded_Y, num_classes=len(y_encoder.classes_))

        else:
            encoder.fit(y)
            # print(encoder.classes_)
            # print(encoder.transform(encoder.classes_))
            encoded_Y = encoder.transform(y)
            return to_categorical(encoded_Y), encoder

def manipulate_training_data(X, Y, subSeqLength, number_subsequences):
    # Todo include Y inflation
    pool = ThreadPool(multiprocessing.cpu_count())

    def make_manipulation(sample):
        if len(sample) > subSeqLength:
            X_train_manipulated = []

            for i in range(number_subsequences):
                start = random.randint(0, len(sample) - subSeqLength)
                subSeq = sample[start:start + subSeqLength]
                X_train_manipulated.append(subSeq)

            return np.array(X_train_manipulated)
        else:
            return

    X_train_manipulated_total = pool.map(make_manipulation, X)
    X_train_manipulated_total = np.array(X_train_manipulated_total)
    shape = X_train_manipulated_total.shape
    X_train_manipulated_total = X_train_manipulated_total.reshape(
        (len(X) * number_subsequences, shape[2], shape[3]))

    y = []
    for i in Y:
        y.append(number_subsequences * [i])
    Y = np.array(y).flatten()
    pool.close()
    pool.join()

    return X_train_manipulated_total, Y




