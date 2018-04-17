import itertools
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.rcParams['backend'] = 'Agg'
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import re
import matplotlib.patches as mpatches

test_pred = np.genfromtxt('/home/go96bix/projects/nanocomb/nanocomb/' + 'pred_vector.csv', delimiter=',')
Y_test = pd.read_csv('/home/go96bix/projects/nanocomb/nanocomb/' + '/Y_test.csv', delimiter='\t', dtype='str', header=None)[1].as_matrix()
X_test = pd.read_csv('/home/go96bix/projects/nanocomb/nanocomb/' + '/X_test.csv', delimiter='\t', dtype='str', header=None)[1].as_matrix()

train_pred = np.genfromtxt('/home/go96bix/projects/nanocomb/nanocomb/' + 'pred_vector_train.csv', delimiter=',')
Y_train = pd.read_csv('/home/go96bix/projects/nanocomb/nanocomb/' + '/Y_train.csv', delimiter='\t', dtype='str', header=None)[1].as_matrix()
X_train = pd.read_csv('/home/go96bix/projects/nanocomb/nanocomb/' + '/X_train.csv', delimiter='\t', dtype='str', header=None)[1].as_matrix()

# test_pred = train_pred
X_test = pd.read_csv('/home/go96bix/projects/nanocomb/nanocomb/' + '/PCA_X_test.csv', delimiter='\t', dtype='str', header=None)[1].as_matrix()
Y_test = pd.read_csv('/home/go96bix/projects/nanocomb/nanocomb/' + '/PCA_Y_test.csv', delimiter='\t', dtype='str', header=None)[1].as_matrix()

def count_codons(input):
    nt = ["A", "C", "G", "T"]
    d_codon = {"".join(list(codon)): idx for idx, codon in enumerate(itertools.product(nt, nt, nt))}
    print(d_codon)
    usage = np.zeros((len(input), 64))

    for idx, seq in enumerate(input):
        seq = re.sub(r"[^ACGTacgt]+", '', seq)
        for i in range(0, len(seq) - 2):
            usage[idx, d_codon[seq[i:i + 3]]] += 1
        usage[idx]=usage[idx]/sum(usage[idx])

    return usage

codon_usage = count_codons(X_test)

"""
1. get all data
2. make full tax as host string
3. sort labels alphabetically
4. print pca
"""

"""adapt data"""
encoder = LabelEncoder()
encoder.fit(Y_test)
print(encoder.classes_)
print(encoder.transform(encoder.classes_))
encoded_Y = encoder.transform(Y_test)
perfect_input = to_categorical(encoded_Y)
# print(encoded_Y)
# species = ['Drosophila melanogaster','Homo sapiens','Mus musculus','Culex pipiens','Cyanocitta cristata','Dermacentor andersoni','Desmodus rotundus','Salmo salar','Zea mays']
# # species = ['Drosophila melanogaster','Drosophila immigrans','Homo sapiens','Mus musculus','Rattus norvegicus','Culex pipiens','Cyanocitta cristata','Dermacentor andersoni','Desmodus rotundus','Salmo salar','Zea mays']
# # species = ['Drosophila melanogaster','Drosophila immigrans','Mus musculus','Felis catus','Homo sapiens','Sus scrofa']
# encoded_species_interest = encoder.transform(species)
# print(encoded_species_interest)

# Mensch
human = "/cellular_organisms/Eukaryota/Opisthokonta/Metazoa/Eumetazoa/Bilateria/Deuterostomia/Chordata/Craniata/Vertebrata/Gnathostomata/Teleostomi/Euteleostomi/Sarcopterygii/Dipnotetrapodomorpha/Tetrapoda/Amniota/Mammalia/Theria/Eutheria/Boreoeutheria/Euarchontoglires/Primates/Haplorrhini/Simiiformes/Catarrhini/Hominoidea/Hominidae/Homininae/Homo/Homo_sapiens"
# Hausschwein
swine = "/cellular_organisms/Eukaryota/Opisthokonta/Metazoa/Eumetazoa/Bilateria/Deuterostomia/Chordata/Craniata/Vertebrata/Gnathostomata/Teleostomi/Euteleostomi/Sarcopterygii/Dipnotetrapodomorpha/Tetrapoda/Amniota/Mammalia/Theria/Eutheria/Boreoeutheria/Laurasiatheria/Cetartiodactyla/Suina/Suidae/Sus/Sus_scrofa/Sus_scrofa_domesticus"
# Huhn
chicken = "/cellular_organisms/Eukaryota/Opisthokonta/Metazoa/Eumetazoa/Bilateria/Deuterostomia/Chordata/Craniata/Vertebrata/Gnathostomata/Teleostomi/Euteleostomi/Sarcopterygii/Dipnotetrapodomorpha/Tetrapoda/Amniota/Sauropsida/Sauria/Archelosauria/Archosauria/Dinosauria/Saurischia/Theropoda/Coelurosauria/Aves/Neognathae/Galloanserae/Galliformes/Phasianidae/Phasianinae/Gallus/Gallus_gallus"
species = [human,swine,chicken]

def multi_plot_pca(species):

    pre_sub_Y_test = [True if x in species else False for x in Y_test]
    sub_Y_test = Y_test[pre_sub_Y_test]
    encoder = LabelEncoder()
    encoder.fit(species)
    print(encoder.classes_)
    print(encoder.transform(encoder.classes_))
    encoded_sub_Y = encoder.transform(sub_Y_test)

    sub_codon_usage = codon_usage[pre_sub_Y_test]

    pca = PCA()
    projected = pca.fit_transform(sub_codon_usage)
    colors = ["steelblue", "saddlebrown", "seagreen", "mediumorchid"]

    def plot_pca(projected, color='grey'):
        """2D Plot"""

        if color == 'grey':
            print('grey')
            plt.scatter(projected[:, 0], projected[:, 1],
                         edgecolor='none', alpha=0.5, c='grey')

        else:
            print('color')
            print(len(projected[:, 0]))
            print(projected[:, 0])
            print(len(projected[:, 1]))
            print(projected[:, 1])
            plt.scatter(projected[:, 0], projected[:, 1],
                        edgecolor='none', alpha=0.5, c=color,
                        cmap=plt.cm.get_cmap('rainbow', len(encoder.classes_)))

    for index, sub_class in enumerate(encoder.transform(encoder.classes_)):
        # species_grey = species.pop[index]

        # pre_species_color = [True if x == sub_class else False for x in encoded_sub_Y]
        # print(encoded_sub_Y[0])
        # pre_species_color = [a for a in encoded_sub_Y if a == sub_class]
        pre_species_color = [encoded_sub_Y==sub_class]
        print("pre species color")
        print(pre_species_color)
        species_color_projected = projected[pre_species_color]
        print("species color projected")
        print(species_color_projected)
        # print(len(projected))
        # print(projected)
        plot_pca(projected)
        # print(len(species_color_projected))
        # print(species_color_projected)
        # das ist nicht das richtige was ich übergebe

        plot_pca(species_color_projected,color=colors[index])



        plt.title('PCA with codon usage')
        handles = []
        for index, label in enumerate(encoder.classes_):
            patch = mpatches.Patch(color=plt.cm.get_cmap('rainbow', len(encoder.classes_))(index / len(encoder.classes_)),
                                   label=label)
            handles.append(patch)
        # plt.legend(handles= handles, loc='best', shadow=False, scatterpoints=1)

        plt.xlabel('component 1')
        plt.ylabel('component 2')
        # plt.colorbar();
        name = encoder.classes_[sub_class].split("/")[-1]
        plt.savefig("PCA_"+str(name)+".png")
        plt.savefig("PCA_" + str(name) + ".pdf")
        plt.clf()
    exit()


multi_plot_pca(species)
exit()

# species = [human]

# #select only rows of species of interest
# test_pred = test_pred[:,encoded_species_interest]

pre_sub_Y_test = [True if x in species else False for x in Y_test]
sub_Y_test = Y_test[pre_sub_Y_test]
encoder = LabelEncoder()
encoder.fit(species)
print(encoder.classes_)
print(encoder.transform(encoder.classes_))
encoded_sub_Y = encoder.transform(sub_Y_test)

# sub_test_pred = test_pred[pre_sub_Y_test]
sub_codon_usage = codon_usage[pre_sub_Y_test]
# sub_perfect_input = perfect_input[pre_sub_Y_test]


# sub_codon_usage = codon_usage

"""2D Plot"""
pca = PCA()
# pca = PCA().fit(sub_test_pred)  # project from 64 to 2 dimensions
# projected = pca.fit_transform(test_pred)
# pca.fit(sub_test_pred)
# print(pca.explained_variance_ratio_)
# projected = pca.fit_transform(sub_test_pred)
projected = pca.fit_transform(sub_codon_usage)
# projected = pca.fit_transform(sub_perfect_input)

plt.scatter(projected[:, 0], projected[:, 1],
            c=encoded_sub_Y, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('rainbow', len(encoder.classes_)))

plt.title('PCA with codon usage')
handles = []
for index, label in enumerate(encoder.classes_):
    patch = mpatches.Patch(color=plt.cm.get_cmap('rainbow', len(encoder.classes_))(index/len(encoder.classes_)), label=label)
    handles.append(patch)
# plt.legend(handles= handles, loc='best', shadow=False, scatterpoints=1)

plt.xlabel('component 1')
plt.ylabel('component 2')
# plt.colorbar();
plt.savefig("PCA_human.pdf")
plt.savefig("PCA_human.png")
exit()
plt.show()

"""3D plot"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(projected[:,0], projected[:,1], projected[:,2], c=encoded_sub_Y, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('rainbow', len(encoder.classes_)))

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
