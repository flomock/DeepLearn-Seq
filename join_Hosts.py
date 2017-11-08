#!/home/go96bix/my/programs/Python-3.6.1/bin/python3.6
import os
import pandas as pd
import json



dic = {"chicken" : "avian", "duck" : "avian", "pigeon" : "avian", "bird" : "avian", "macaque" : "monkey",
       "sparrow": "avian", "turkey": "avian", "goose": "avian", "gull": "avian", "owl": "avian", "parrot": "avian",
       "woodpecker": "avian", "seabird": "avian", "penguin": "avian", "blackbird": "avian", "crow": "avian",
       "dove": "avian", "finch": "avian", "falcon": "avian", "flamingo": "avian", "hawk": "avian",
       "mockingbird": "avian", "raven": "avian","american flamingo": "avian", "swan": "avian","pig": "swine"}

mypath = '/home/go96bix/Data_virus_host_predict'
for root, dirs, files in os.walk(mypath):
    # print root
    for filename in [f for f in files if f.endswith(".json") and not f.endswith("_new.json")]:
        print(filename)

        address = str(os.path.join(root, filename))
        # regex = re.compile(">.*\n")
        # inputfile = pd.read_json(address)


            # pprint(data)
        with open(address) as f:
            names=[]
            text = f.readlines()
            # get ids from json file
            for line in text:
                data = json.loads(line)
                names.append(data['name'])
            # load table
            tsv_address = str(os.path.join(root,filename[:-5]+".tsv"))
            tsv_dataframe = pd.DataFrame.from_csv(tsv_address,header=0,sep="\t")

            # stackoverflow 13757090
            tsv_dataframe.columns = [c.replace('GenBank Accession', 'id') for c in tsv_dataframe.columns]
            tsv_dataframe.columns = [c.replace('Sequence Accession', 'id') for c in tsv_dataframe.columns]
            tsv_dataframe.columns = [c.replace('Host Species', 'Host') for c in tsv_dataframe.columns]
            # remove * from ids like NC_02314*
            tsv_dataframe.id = tsv_dataframe.id.str.replace('\*', '')
            tsv_dataframe.Host = tsv_dataframe.Host.str.replace('IRD:', '')

            # make an dict of the table
            # stackoverflow 26716616
            tsv_dataframe.set_index("id", drop=True, inplace=True)
            id_dict = tsv_dataframe.to_dict(orient="index")
            # get the new host and save changes in new json
            new_json = []
            for i in range(len(names)):
                line = text[i]
                name = names[i]
                data = json.loads(line)
                # print(data['host'])
                try:
                    new_host= id_dict[name]['Host']
                    species = new_host.split('/')
                    if (len(species) > 1):
                        species = species[1]
                    else:
                        species = species[0]
                    try:
                        species = dic[species.lower()]
                    except:
                        # print(species)
                        pass
                    # print(species.lower())
                    data['host'] = species.lower()
                except:
                    print("cannot find "+ name)
                    continue
                # print(data['host'])
                new_json.append(json.dumps(data))
            json_address = str(os.path.join(root,filename[:-5]+"_new.json"))
            new_file = open(json_address,'w')
            for line in new_json:
                new_file.write("%s\n" % line)