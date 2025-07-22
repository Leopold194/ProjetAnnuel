# test_upload_images.py

from pathlib import Path
from pymongo import MongoClient
import gridfs
from lib_gridfs import init_db_fs, get_images  # assuming this is your file

def put_first_n_images(folder_path: str, n: int = 15,
                       overwrite: bool = True,
                       uri: str = "mongodb://localhost:27017",
                       db_name: str = "affiches"):

    db, fs = init_db_fs(uri, db_name)
    all_images = get_images(folder_path)
    print(f"Found {len(all_images)} image(s)")

    for filename, data in all_images[:n]:
        existing = db.fs.files.find_one({"filename": filename})
        if existing and overwrite:
            fs.delete(existing["_id"])
            fs.put(data, filename=filename)
            print(f"[OVERWRITTEN] {filename}")
        elif not existing:
            fs.put(data, filename=filename)
            print(f"[INSERTED] {filename}")
        else:
            print(f"[SKIPPED] {filename}")

if __name__ == "__main__":
    folder = r"C:\Users\AIO\Documents\ESGI\2024-2025\ProjetAnnuel\dataset\posters_20x30"
    put_first_n_images(folder, n=15)
