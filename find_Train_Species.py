'''get fasta, print must common species in file.
For use as train and test species'''

import re
import os
import operator

###MY STUFF
'''take fasta file and make dict with keys species and value how often represented in data'''
# mypath = '/home/florian/Dropbox/Masterarbeit/data'
# mypath = '/home/florian/Dropbox/Masterarbeit/Docker-projects/temp/'
mypath = '/home/go96bix/Data_virus_host_predict'
dic = {"chicken" : "avian", "duck" : "avian", "pigeon" : "avian", "bird" : "avian", "macaque" : "monkey",
       "sparrow": "avian", "turkey": "avian", "goose": "avian", "gull": "avian", "owl": "avian", "parrot": "avian",
       "woodpecker": "avian", "seabird": "avian", "penguin": "avian", "blackbird": "avian", "crow": "avian",
       "dove": "avian", "finch": "avian", "falcon": "avian", "flamingo": "avian", "hawk": "avian",
       "mockingbird": "avian", "raven": "avian","american flamingo": "avian", "swan": "avian","pig": "swine"}
# sparrow turkey goose gull owl parrot woodpecker seabird penguin blackbird crow dove finch falcon flamingo hawk mockingbird raven swan
# pig
for root, dirs, files in os.walk(mypath):
    # print root

    for filename in [f for f in files if f.endswith("dump.fasta")]:
        print(filename)
        address = str(os.path.join(root, filename))
        regex = re.compile(">.*\n")
        with open(address) as f:
            text = f.read()
            headers = regex.findall(text)
            number_animals = {}
            for head in headers:
                #get species without \n and makes all lowercase
                species_detailed = head.split('|')[1][:].lower()
                species = species_detailed.split(';'or ',')[0]
                # species_all = species
                # species = species_all.split('/')
                # if (len(species)>1):
                #     species = species[1]
                # else:
                #     species = species[0]
                # try:
                #     species = dic[species]
                # except:
                #     pass
                # else:
                # if (species=="chicken")
                if not species in number_animals:
                    number_animals.update({str(species):1})
                else:
                    number_animals[str(species)]+=1

        '''stackoverflow.com/questions/613183'''
        sorted_animals = sorted(number_animals.items(),key=operator.itemgetter(1))

        for i in sorted_animals:
            print(i)
    exit()