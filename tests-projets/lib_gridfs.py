import requests
from PIL import Image
from io import BytesIO
import os
from pymongo import MongoClient
import gridfs
import json
from tqdm import tqdm
import numpy as np
import datetime

# 1) Connexion
def get_db(uri: str = "mongodb://localhost:27017", db_name="movies") -> "pymongo.database.Database":
    client = MongoClient(uri)
    return client[db_name]

# 2) Initialisation de GridFS (par défaut bucket "fs")
def get_fs(db) -> gridfs.GridFS:
    return gridfs.GridFS(db)

# 3) Wrapper pour db et fs
def init_db_fs(uri: str = "mongodb://localhost:27017", db_name="movies") -> tuple:
    db = get_db(uri, db_name)
    fs = get_fs(db)
    return db, fs

from concurrent.futures import ThreadPoolExecutor, as_completed

def upload_single_poster(entry, fs):
    url = entry["poster"]
    genre = entry["genre"]
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert("RGB")
        width, height = img.size

        fs.put(
            BytesIO(response.content),
            filename=url.split("/")[-1],
            metadata={
                "genre": genre,
                "fullsize": True,
                "size": {"width": width, "height": height}
            }
        )
        # print(f" Uploaded: {url} ({width}x{height})")
        return None  # Pas d'erreur
    except Exception as e:
        print(f"Failed to upload {url}: {e}")
        return url  # Erreur → retourner l'URL

def upload_posters_from_json_parallel(json_path, mongo_uri="mongodb://localhost:27017", db_name="movies", bucket_name="fs", max_workers=10):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db, collection=bucket_name)
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    failed_images = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(upload_single_poster, entry, fs) for entry in data]
        for future in as_completed(futures):
            result = future.result()
            if result:  # Si une URL est retournée, c'est une erreur
                failed_images.append(result)

    with open("errors.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(failed_images))

    return failed_images

# Récupérer une image par son ObjectId
def get_image(image_id, uri: str = "mongodb://localhost:27017", db_name="movies") -> bytes:
    _, fs = init_db_fs(uri, db_name)
    try:
        return fs.get(image_id).read()
    except gridfs.errors.NoFile:
        print(f"No file found with id {image_id}")
        return None

# Supprimer une image par son ObjectId
def delete_image(image_id, uri: str = "mongodb://localhost:27017", db_name="movies"):
    _, fs = init_db_fs(uri, db_name)
    try:
        fs.delete(image_id)
        print(f"Deleted image {image_id}")
    except Exception as e:
        print(f"Error deleting {image_id}: {e}")

# Lister tous les ObjectId présents dans GridFS
def get_file_ids(uri: str = "mongodb://localhost:27017", db_name="movies") -> list:
    db, _ = init_db_fs(uri, db_name)
    return [doc["_id"] for doc in db.fs.files.find({}, {"_id": 1})]

def load_posters_with_size(size: tuple, genres: list = None, mongo_uri="mongodb://localhost:27017", db_name="movies", bucket_name="fs"):
    """
    Charge toutes les posters avec la taille spécifiée depuis GridFS.
    Si elles n'existent pas, redimensionne les fullsize, les stocke en base et les retourne.
    """

    width, height = size
    client = MongoClient(mongo_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db, collection=bucket_name)

    if genres is None:
        matching_files = list(db[f"{bucket_name}.files"].find({"metadata.size.width": width, "metadata.size.height": height}))
    else:
        matching_files = []
        for g in genres:
            matching_files.extend(list(db[f"{bucket_name}.files"].find({"metadata.size.width": width, "metadata.size.height": height, "metadata.genre": g})))

    images = []

    if matching_files:
        genres = []
        print(f"📦 Found {len(matching_files)} images of size {width}x{height}")
        for doc in tqdm(matching_files, desc="📥 Loading resized images"):
            try:
                image_data = fs.get(doc["_id"]).read()
                
                image = Image.open(BytesIO(image_data)).convert("RGB")
                image_array = np.array(image)
                flat_array = image_array.flatten()
                images.append(flat_array)
                
                genres.append(doc["metadata"].get("genre", "Unknown"))
            except gridfs.errors.NoFile:
                print(f"❌ Missing file for {doc['_id']}")
    else:
        print(f"🛠 No resized images found : resizing fullsize images to {width}x{height}...")
        if genres is None:
            fullsize_files = list(db[f"{bucket_name}.files"].find({"metadata.fullsize": True}))
        else:
            fullsize_files=[]
            for g in genres:
                fullsize_files.extend(list(db[f"{bucket_name}.files"].find({"metadata.fullsize":True, "metadata.genre": g})))             
        print(len(fullsize_files))
        for doc in tqdm(fullsize_files, desc="🛠 Resizing and saving"):
            try:
                original = fs.get(doc["_id"]).read()
                genre = doc["metadata"].get("genre", "Unknown")

                img = Image.open(BytesIO(original)).convert("RGB")
                resized_img = img.resize((width, height))
                buf = BytesIO()
                resized_img.save(buf, format="JPEG")
                buf.seek(0)

                fs.put(buf, filename=doc["filename"], metadata={"genre": genre, "fullsize": False, "size": {"width": width, "height": height}})

                image_array = np.array(resized_img)
                flat_array = image_array.flatten()
                
                images.append(flat_array)
                genres.append(genre)
            except Exception as e:
                print(f"❌ Error processing {doc['filename']}: {e}")

    return images, genres

def load_posters_with_size(size: tuple, genres: list = None, mongo_uri="mongodb://localhost:27017", db_name="movies", bucket_name="fs"):
    """
    Charge toutes les posters avec la taille spécifiée depuis GridFS.
    Si certaines sont manquantes, redimensionne les fullsize correspondantes, les stocke et les retourne.
    """
    width, height = size
    client = MongoClient(mongo_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db, collection=bucket_name)

    # Step 1: Find existing resized images
    resized_query = {"metadata.size.width": width, "metadata.size.height": height}
    if genres:
        resized_query["metadata.genre"] = {"$in": genres}
    resized_files = list(db[f"{bucket_name}.files"].find(resized_query))
    resized_filenames = set(doc["filename"] for doc in resized_files)

    # Step 2: Find all fullsize images
    fullsize_query = {"metadata.fullsize": True}
    if genres:
        fullsize_query["metadata.genre"] = {"$in": genres}
    fullsize_files = list(db[f"{bucket_name}.files"].find(fullsize_query))

    images = []
    result_genres = []

    # Step 3: Load already resized images
    print(f"📦 Found {len(resized_files)} resized images of size {width}x{height}")
    for doc in tqdm(resized_files, desc="📥 Loading resized images"):
        try:
            image_data = fs.get(doc["_id"]).read()
            image = Image.open(BytesIO(image_data)).convert("RGB")
            flat_array = np.array(image).flatten()
            images.append(flat_array)
            result_genres.append(doc["metadata"].get("genre", "Unknown"))
        except gridfs.errors.NoFile:
            print(f"❌ Missing file for {doc['_id']}")

    # Step 4: Resize missing images
    to_resize = [doc for doc in fullsize_files if doc["filename"] not in resized_filenames]
    print(f"🛠 Found {len(to_resize)} fullsize images to resize to {width}x{height}")
    for doc in tqdm(to_resize, desc="🛠 Resizing and saving missing images"):
        try:
            original = fs.get(doc["_id"]).read()
            genre = doc["metadata"].get("genre", "Unknown")
            img = Image.open(BytesIO(original)).convert("RGB")
            resized_img = img.resize((width, height))

            buf = BytesIO()
            resized_img.save(buf, format="JPEG")
            buf.seek(0)

            fs.put(buf, filename=doc["filename"], metadata={"genre": genre, "fullsize": False, "size": {"width": width, "height": height}})

            flat_array = np.array(resized_img).flatten()
            images.append(flat_array)
            result_genres.append(genre)
        except Exception as e:
            print(f"❌ Error processing {doc['filename']}: {e}")

    return images, result_genres

def stock_metrics(train_loss: list, test_loss: list, train_accuracy: list, test_accuracy: list, model_params: dict, model: str, mongo_uri="mongodb://localhost:27017", db_name="movies"):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    coll = db["metrics"]
    
    d = {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "model": model,
        "timestamp": datetime.datetime.now()
    }
    
    for p in model_params.keys():
        d[p] = model_params[p]
    
    result = coll.insert_one(d)
    return result

if __name__ == "__main__":
    file_path = r"C:\Users\leopo\Documents\ESGI\3rd_year\ProjetAnnuel\dataset\movies_full_classif.json"
    failed_images = upload_posters_from_json_parallel(file_path, max_workers=40)
