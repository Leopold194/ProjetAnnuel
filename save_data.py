from pymongo import MongoClient
import json

def get_database():
    CONNECTION_STRING = "mongodb://localhost:27017/"
    client = MongoClient(CONNECTION_STRING)
    return client['projetannuel']

if __name__ == "__main__":   
    dbname = get_database()
    collection = dbname["posters"]

    with open("dataset/images_flat_full.json", "r", encoding="utf-8") as dataset:
        data = json.load(dataset)

    print("Fichier JSON chargé !")

    result = collection.insert_many(data)
    print("Données insérées dans MongoDB !")