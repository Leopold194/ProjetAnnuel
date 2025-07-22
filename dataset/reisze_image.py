import os
from PIL import Image
from tqdm import tqdm

def resize_folder(
    src_folder: str,
    dst_folder: str,
    size: tuple[int, int],
    overwrite: bool = False,
    formats: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")
):
    """
    Resize every image in src_folder to `size` (width, height) and write to dst_folder.
    
    - overwrite: if False, skips files already existing in dst_folder.
    - formats: tuple of allowed file extensions (lowercase).
    """
    os.makedirs(dst_folder, exist_ok=True)
    w, h = size

    for fname in tqdm(os.listdir(src_folder), desc="Resizing images"):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in formats:
            continue

        src_path = os.path.join(src_folder, fname)
        dst_path = os.path.join(dst_folder, fname)

        if not overwrite and os.path.exists(dst_path):
            continue

        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img_resized = img.resize((w, h), Image.LANCZOS)
                img_resized.save(dst_path, format="JPEG")
        except Exception as e:
            tqdm.write(f"⚠️ Failed to resize {fname}: {e}")

    print(f"✅ Done. Resized images saved in: {dst_folder}")

if __name__ == "__main__":
    # --- User configuration ---
    SRC_DIR = r"C:\Users\AIO\Documents\ESGI\2024-2025\ProjetAnnuel\dataset\raw_posters_full"
    DST_DIR = r"C:\Users\AIO\Documents\ESGI\2024-2025\ProjetAnnuel\dataset\posters_20x30"
    TARGET_SIZE = (30, 30)   # (width, height)
    OVERWRITE_EXISTING = False

    resize_folder(SRC_DIR, DST_DIR, TARGET_SIZE, overwrite=OVERWRITE_EXISTING)
