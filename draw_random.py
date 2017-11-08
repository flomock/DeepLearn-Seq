"""In: collection, match dictionary (i.e. the filter), sample size n
Out: documents of samples as database cursor (i.e. generator)

Usage:
from pymongo import MongoClient

client = MongoClient("localhost:27017")
db = client["zoo"]
collection = db.get_collection('influenza_a_virus')

match = {
    'annotation.name': 'HA',
    'metadata.host': 'Avian',
    'metadata.date.y': {'$gte': 2000}
}

gen = random_draw(collection, match, n)
next(gen)
# GQ377055"""

def draw_random(collection, match, n):
    name = collection.name
    project = {'$project': {
        '_id': 1,  # TODO vll noch später entfernen oder host anstelle von _id nehmen
        "host": 1,
        "seq": 1,
        "length": {'$strLenCP': "$seq"},
        "collection_name": name
    }}
    sample = {'size': n}
    pipeline = [project,{'$match': match}, {'$sample': sample}]
    query = collection.aggregate(pipeline,allowDiskUse=True)
    return query
