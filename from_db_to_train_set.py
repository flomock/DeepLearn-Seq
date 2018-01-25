#!/home/go96bix/my/programs/Python-3.6.1/bin/python3.6

from collections import Counter
from itertools import chain
import random
import datetime
import numpy as np
import os
import re
import pandas as pd
# import progressbar
from pymongo import MongoClient
from skbio import TabularMSA, DNA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.externals import joblib
# import sys
# from xgboost import XGBClassifier


# from zoo.utils import iter_seq, to_fasta
from zoo.align import msa_subset
from zoo.learn import entropy_select

# sys.path.append('/Users/pi/commonplace/compbio/protocols/utils')
# from draw_random import draw_random
from draw_random import draw_random
from one_hot_encoding import ohe_matrix
# from shannon import shannon
# from chunk import chunk
# from utils import iter_seq, to_fasta
# from subprocess import call, check_output
import random

# import datetime
'''
0. Database connection, functions, constants etc.
'''

suffixold = '2017-06-23T14:37:54.637627'
suffixnew = str(datetime.datetime.now().isoformat())
directory = '/home/go96bix/Dropbox/Masterarbeit/data/machineLearning/' + suffixnew
makeMafft = True
makeoutput = True

# client = MongoClient()
# db = client["multiV"]

# print(list(collection.find({'host': {'$regex': '^homo sapiens'}}))[0])
# HOSTS = 'mallard, chicken, Homo sapiens'.split(', ') #homosapiens always with aditional infos
# HOSTS = 'Macaca, Rousettus, Homo sapiens'.split(', ') #homosapiens always with aditional infos
HOSTS = 'human, swine, avian'.split(', ')  # homosapiens always with aditional infos
SIZE = 0.33
SEED = 42


def get_Species_from_DB(species=['all'],  mongoclient="localhost:27017", zoo="multiV"):
    """

    :param species:
    :param mongoclient:
    :param zoo:
    :return:
    """
    client = MongoClient(mongoclient)
    db = client[zoo]
    if species == ['all']:
        species = db.collection_names()
    allCells = []
    for cell in species:
        collection = db.get_collection(cell)
        allCells.append(collection)
    return allCells


def stratify_via(stratify_feature, species,min_Seq_length = 1000, use_subseq=True, subseq_length = 1000, stepsize = 5, limit_n_samples=False):
    """

    :param stratify_feature:
    :param species:
    :param min_Seq_length:
    :param use_subseq:
    :param subseq_length:
    :param stepsize:
    :param limit_n_samples:
    :return:
    """
    # TODO implement taxonomy feature
    # "taxonomy": ["Viruses", "ssRNA viruses", "ssRNA positive-strand viruses, no DNA stage",
    # "Nidovirales", "Coronaviridae", "Coronavirinae", "Gammacoronavirus"]
    global SEED
    global SIZE
    minSample = []
    minSampleTrain = []
    minSampleTest = []
    TESTS = []
    TRAINS = []

    if stratify_feature == "host":
        for cell in species:
            project = {'$project': {
                '_id': '$host',  # vll noch später entfernen oder host anstelle von _id nehmen
                "host": 1,
                "seq": 1,
                "length": {'$strLenCP': "$seq"}
            }}
            if min_Seq_length:
                match = {'$match': {
                    'host': {'$in': [re.compile(host) for host in HOSTS]},
                    'length': {'$gte': min_Seq_length}
                }}
            else:
                match = {'$match': {
                    'host': {'$in': [re.compile(host) for host in HOSTS]},
                }}
            # here the difference for stratify
            group = {'$group': {
                '_id': '$host',
                'cnt': {'$sum': 1},
            }}
            pipeline = [project, match, group]
            q = cell.aggregate(pipeline)
            dicNumberHosts = {}
            for i in q:
                # generates a Dictionary with short Host names and number of counts
                for h in HOSTS:
                    if str(h) in str(i['_id']):
                        cntHost = dicNumberHosts.get(h, 0) + i['cnt']
                        dicNumberHosts[h] = cntHost
                        break
            n = min(dicNumberHosts.values())  # for a balanced sample
            minSample.append(n)
        overallMin = min(minSample)

        for cell in species:
            l = []
            for h in HOSTS:
                h = re.compile(h)
                match = {
                    'host': h
                    # 'host': {'$in': [re.compile(host) for host in HOSTS]},
                }
                query = draw_random(cell, match, overallMin)

                l.append(query)
            query = chain(*l)
            gen = ((i['_id'], i['host'], i['seq'], i['length']) for i in query)
            df = pd.DataFrame.from_records(gen, columns=['id', 'host', 'seq', 'length'])
            test = pd.DataFrame()
            for h in HOSTS:
                df_host = df[df.host == h]
                test_sub = df_host.sample(frac=SIZE, random_state=SEED)
                test = test.append(test_sub)
            # TESTS = [[species[host[samples(df)]]]
            train = df.drop(test.index)

            count_train = {}
            count_test = {}

            host_df_array_train = []
            host_df_array_test = []
            for h in HOSTS:
                test_host = test[test.host == h]
                train_host = train[train.host == h]
                host_df_array_test.append(test_host)
                host_df_array_train.append(train_host)

                if use_subseq:
                    for element in test_host.length:
                        cntSnips = int((element - subseq_length) / stepsize) + 1
                        count_test[h] = count_test.get(h, 0) + cntSnips

                    for element in train_host.length:
                        cntSnips = int((element - subseq_length) / stepsize) + 1
                        count_train[h] = count_train.get(h, 0) + cntSnips
            TESTS.append(host_df_array_test)
            TRAINS.append(host_df_array_train)

            n = min(count_train.values())  # for a balanced sample
            m = min(count_test.values())  # for a balanced sample
            minSampleTrain.append(n)
            minSampleTest.append(m)
        minSampleTest = min(minSampleTest)
        minSampleTrain = min(minSampleTrain)

    elif(stratify_feature=="species"):
        dicNumberSpecies = {}
        for cell in species:
            print(cell.count())
            cntSpecies = dicNumberSpecies.get(cell.name, 0) + cell.count()
            dicNumberSpecies[cell.name] = cntSpecies
        n = min(dicNumberSpecies.values())
        l = []
        for cell in species:
            print(cell.name)
            query = draw_random(cell,{},n)
            l.append(query)
        query = chain(*l)
        gen = ((i['_id'], i['collection_name'], i['seq'], i['length']) for i in query)
        df = pd.DataFrame.from_records(gen, columns=['id', 'species', 'seq', 'length'])
        test = pd.DataFrame()
        for cell in species:
            df_species = df[df.species == cell.name]
            test_sub = df_species.sample(frac=SIZE, random_state=SEED)
            test = test.append(test_sub)
        train = df.drop(test.index)
        count_train = {}
        count_test = {}

        species_df_array_train = []
        species_df_array_test = []
        for cell in species:
            test_species = test[test.species == cell.name]
            train_species = train[train.species == cell.name]
            species_df_array_test.append(test_species)
            species_df_array_train.append(train_species)

            if use_subseq:
                for element in test_species.length:
                    cntSnips = int((element - subseq_length) / stepsize) + 1
                    count_test[cell.name] = count_test.get(cell.name, 0) + cntSnips

                for element in train_species.length:
                    cntSnips = int((element - subseq_length) / stepsize) + 1
                    count_train[cell.name] = count_train.get(cell.name, 0) + cntSnips
        TESTS.append(species_df_array_test)
        TRAINS.append(species_df_array_train)

        minSampleTrain = min(count_train.values())  # for a balanced sample
        minSampleTest = min(count_test.values())  # for a balanced sample

    else:
        pass
    if limit_n_samples != False:
        if (minSampleTest > (limit_n_samples*SIZE) and minSampleTrain > (limit_n_samples*(1-SIZE))):
            minSampleTest = int(limit_n_samples*SIZE)
            minSampleTrain = int(limit_n_samples*(1-SIZE))
        else:
            print("limit_n_samples to big use default value instead")
            exit()
    test_set_df, index_test = cut_motherseq_in_Dataframe(TESTS, minSampleTest, stepsize=stepsize)
    train_set_df, index_train = cut_motherseq_in_Dataframe(TRAINS, minSampleTrain, stepsize=stepsize)
    return test_set_df, train_set_df, index_test, index_train


def cut_motherseq_in_Dataframe(Dataframe_array, Samplesize, stepsize=5, subseq_length=1000):
    """
    receives array with all species, each species consists of different hosts array
    :param Dataframe_array: [species[hosts[samples as dataframe]]]
    :param Samplesize: how many sub-samples drawn
    :return: Set of sub-samples and were to find them in the samples
    """
    #         df = pd.DataFrame.from_records(gen, columns=['id', 'host','seq','length'])
    global HOSTS
    df = pd.DataFrame()
    indexes_used = []
    for species_df in Dataframe_array:
        # df_sub =pd.DataFrame()
        for host_df in species_df:
            df_sub = []
            print("next host!")
            for index, row in host_df.iterrows():
                # motherseq = np.array(list(row.seq))
                motherseq = row.seq
                for startpos in range(0, len(motherseq) - (subseq_length - 1), stepsize):
                    row2 = row
                    row2.seq = motherseq[startpos:startpos + subseq_length]
                    df_sub.append(row2)

            print(len(df_sub))

            df_sub2 = random.sample(df_sub, Samplesize)
            df_help = pd.DataFrame()
            for row in df_sub2:
                df_help = df_help.append(row)
            df = df.append(df_help)
            indexes_used.append(df_help.index)

    return df, indexes_used

def join_seq_to_df(Dataframe_array, max_length):
    """
    receives array with all species, each species consists of different hosts array
    :param Dataframe_array:
    :param max_length: length of longest seq
    :return: set of samples with added N so all have the same length
    """
    print("join_seq_to_df()")
    print("will be implemented if needed")


def save_set(test_set, train_set, index_test, index_train, dir=directory, Y_column = 'host'):
    print(test_set.species.value_counts())
    X_test = test_set.as_matrix(columns=['seq'])

    arr = np.arange(len(X_test))
    np.random.shuffle(arr)
    selection_test = arr
    # exit()
    X_test = np.array(X_test).flatten()
    X_test = [np.array(list(row)) for row in X_test]
    X_test = ohe_matrix(np.array(X_test)[selection_test])

    X_train = train_set.as_matrix(columns=['seq'])
    arr = np.arange(len(X_train))
    np.random.shuffle(arr)
    selection_train = arr

    X_train = np.array(X_train).flatten()
    X_train = [np.array(list(row)) for row in X_train]
    X_train = ohe_matrix(np.array(X_train)[selection_train])

    Y_test = test_set.as_matrix(columns=[Y_column])
    Y_test = Y_test[selection_test]
    Y_train = train_set.as_matrix(columns=[Y_column])
    Y_train = Y_train[selection_train]
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.savetxt(dir + '/X_train.csv', X_train, fmt='%s', delimiter=',')
    np.savetxt(dir + '/X_test.csv', X_test, fmt='%s', delimiter=',')
    np.savetxt(dir + '/Y_train.csv', Y_train, fmt='%s', delimiter=',')
    np.savetxt(dir + '/Y_test.csv', Y_test, fmt='%s', delimiter=',')
    np.savetxt(dir + '/index_test_sample.csv', index_test, fmt='%s', delimiter=',')
    np.savetxt(dir + '/index_train_sample.csv', index_train, fmt='%s', delimiter=',')
    np.savetxt(dir + '/index_test_rearrange.csv', selection_test, fmt='%s', delimiter=',')
    np.savetxt(dir + '/index_train_rearrange.csv', selection_train, fmt='%s', delimiter=',')



species = get_Species_from_DB(species=['flavi', 'corona', 'influ'])
test_set_df, train_set_df, index_test, index_train = stratify_via("species",species,min_Seq_length=1000,stepsize=5,limit_n_samples=10000)
save_set(test_set_df,train_set_df,index_test,index_train,Y_column='species')

# todo VERY INTERESTING pad_sequences https://keras.io/preprocessing/sequence/
# todo use if not wanted all sequnces have same length
# simply makes array with gaps, 0 add end of seq until max length is reached