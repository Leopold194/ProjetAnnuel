# ─── Cell 1: Download & save full-size posters ────────────────────────────────
import os
import json
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Config
RAW_DIR = r"C:\Users\AIO\Documents\ESGI\2024-2025\ProjetAnnuel\dataset\raw_posters_full"
os.makedirs(RAW_DIR, exist_ok=True)
TIMEOUT = 10

# Load your JSON metadata
with open(r"C:\Users\AIO\Documents\ESGI\2024-2025\ProjetAnnuel\dataset\movies_binary_classif.json", "r") as f:
    movies = json.load(f)

# Download + save loop
for idx, mv in enumerate(tqdm(movies, desc="Downloading full-size posters")):
    url = mv.get("poster")
    if not url:
        continue
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        # Derive extension from original format (fallback to jpg)
        ext = (img.format or "JPEG").lower()
        fname = f"poster_{idx:04d}.{ext}"
        img.save(os.path.join(RAW_DIR, fname), format=img.format or "JPEG")
    except Exception as e:
        tqdm.write(f"❌ Failed [{idx}] → {e}")
