import numpy as np
import keras
import os
import multiprocessing.pool
from functools import partial
import keras.preprocessing.image as image
from random import sample as randsomsample
import pandas as pd
import deep_learning_sequences
import DataParsing


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, directory, classes=None, number_subsequences=32, dim=(32, 32, 32), n_channels=6,
                 n_classes=10, shuffle=True, n_samples=None, seed=None, faster=True, online_training=False):
        'Initialization'
        self.directory = directory
        self.classes = classes
        self.dim = dim
        self.labels = None
        self.list_IDs = None
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.seed = seed
        self.online_training = online_training
        if number_subsequences == 1:
            self.shrink_timesteps = False
        else:
            self.shrink_timesteps = True

        self.number_subsequences = number_subsequences

        if faster == True:
            self.faster = 16
        elif type(faster) == int and faster > 0:
            self.faster = faster
        else:
            self.faster = 1

        self.number_samples_per_batch = self.faster

        self.number_samples_per_class_to_pick = n_samples

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
            self.classes = classes

        self.n_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        # want a dict which contains dirs and number usable files
        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(image._count_valid_files_in_directory,
                                   white_list_formats={'csv'},
                                   follow_links=None,
                                   split=None)
        self.samples = pool.map(function_partial, (os.path.join(directory, subdir) for subdir in classes))
        self.samples = dict(zip(classes, self.samples))

        results = []

        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(pool.apply_async(image._list_valid_filenames_in_directory,
                                            (dirpath, {'csv'}, None, self.class_indices, None)))

        self.filename_dict = {}
        for res in results:
            classes, filenames = res.get()
            for index, class_i in enumerate(classes):
                self.filename_dict.update({f"{class_i}_{index}": filenames[index]})

        pool.close()
        pool.join()

        if not n_samples:
            self.number_samples_per_class_to_pick = min(self.samples.values())

        self.on_epoch_end()

    # in images wird ein groesses arr classes gemacht (fuer alle sampels) darin stehen OHE die Class
    # erstelle filename liste in der die zugehoerige file adresse steht
    # laesst sich mergen mit version die oben verlinked

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.number_samples_per_batch))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.number_samples_per_batch:(index + 1) * self.number_samples_per_batch]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, indexes)

        return X, y

    def on_epoch_end(self):
        'make X-train sample list'
        """
        1. go over each class
        2. select randomly #n_sample samples of each class
        3. add selection list to dict with class as key 
        """

        self.class_selection_path = np.array([])
        self.labels = np.array([])
        for class_i in self.classes:
            samples_class_i = randsomsample(range(0, self.samples[class_i]), self.number_samples_per_class_to_pick)
            self.class_selection_path = np.append(self.class_selection_path,
                                                  [self.filename_dict[f"{self.class_indices[class_i]}_{i}"] for i in
                                                   samples_class_i])
            self.labels = np.append(self.labels, [self.class_indices[class_i] for i in samples_class_i])

        self.list_IDs = self.class_selection_path

        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            if self.seed:
                np.random(self.seed)
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, self.dim, self.n_channels),dtype='str')
        X = np.empty((self.number_samples_per_batch), dtype=object)
        Y = np.empty((self.number_samples_per_batch), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # load tsv, parse to numpy array, get str and set as value in X[i]
            X[i] = pd.read_csv(os.path.join(self.directory, ID), delimiter='\t', dtype='str', header=None)[1].values[0]

            # Store class
            Y[i] = self.labels[indexes[i]]

        maxLen = self.number_subsequences * self.dim
        X = DataParsing.encode_string(maxLen=maxLen, x=X, y=[], y_encoder=None, repeat=True, use_spacer=True,
                                      online_Xtrain_set=False)

        assert self.shrink_timesteps != True or self.online_training != True, "online_training shrinks automatically " \
                                                                              "the files, please deactivate shrink_timesteps"

        if self.shrink_timesteps:
            X, Y = deep_learning_sequences.shrink_timesteps(input_subSeqlength=self.dim, X=X, Y=Y)
            # pass
        # Todo zur zeit sind noch alle subsamples in einem batch
        elif self.online_training:
            X, Y = DataParsing.manipulate_training_data(X=X, Y=Y, subSeqLength=self.dim,
                                                     number_subsequences=self.number_subsequences)
        return X, keras.utils.to_categorical(Y, num_classes=self.n_classes)
