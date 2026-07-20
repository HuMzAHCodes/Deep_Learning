# Uploading Large Datasets to Google Colab — Reliable Method

Your tutor's method (`kaggle datasets download` via `kaggle.json`) is outdated and now
commonly fails due to Kaggle's newer token system, phone/ID verification requirements,
and per-dataset consent flags. Use this Drive-based method instead — it's more reliable
and avoids Kaggle's API entirely.

## Why not upload directly to Colab?
Colab's direct file upload goes over your home internet's **upload speed**, which is
usually much slower than download speed. A 1GB+ file can take an hour or more this way,
and it often gets silently cut off (leading to `BadZipFile` errors from a truncated zip).

## Why not use the Kaggle API?
Kaggle now requires phone verification (and sometimes stricter identity checks) before
allowing API downloads — even for public datasets. This makes `kaggle datasets download`
unreliable for many accounts.

## The reliable method: PC → Google Drive → Colab

### Step 1: Upload the zip to Google Drive (from your PC)
- Go to [drive.google.com](https://drive.google.com)
- Upload the dataset zip file there
- Drive's uploader is more resilient to interruptions than Colab's direct upload
- Let it fully finish — check the file size after upload to confirm it matches your original

### Step 2: Mount Google Drive in Colab
```python
from google.colab import drive
drive.mount('/content/drive')
```
This asks you to authenticate once per runtime session. Every time you switch runtime
type (e.g., CPU → GPU) or the runtime resets, you'll need to re-run this.

### Step 3: Find the exact file path
```python
import os
for f in os.listdir('/content/drive/MyDrive'):
    print(f)
```
Confirm the exact filename and whether it's in `MyDrive` directly or inside a subfolder
(e.g., `Colab Notebooks/`).

### Step 4: Extract the zip into Colab's local disk
```python
import zipfile

with zipfile.ZipFile('/content/drive/MyDrive/your_file.zip', 'r') as zip_ref:
    zip_ref.extractall('/content')
```
Extracting to local Colab disk (`/content`) rather than reading directly from Drive
gives much faster read speeds during model training.

### Step 5: Verify the extracted folder structure
```python
import os
for root, dirs, files_ in os.walk('/content'):
    print(root, '->', len(files_), 'files')
```
This confirms the folders/files extracted correctly and shows you the exact paths
to use when loading data (e.g., with `image_dataset_from_directory`).

## Notes on runtime switching
Switching between CPU/GPU/TPU runtimes wipes the Colab machine's local storage
(`/content/`), so any extracted files are lost. You'll need to re-run Steps 2–5
after any runtime change — Drive itself stays untouched, so you never need to
re-upload the zip.

## Quick troubleshooting checklist
| Symptom | Likely Cause |
|---|---|
| `BadZipFile: File is not a zip file` | Upload was incomplete/corrupted — check file size matches original |
| `FileNotFoundError` on extract | Wrong path/filename — re-check with `os.listdir` |
| Kaggle API `403 Forbidden` | Token/auth issue or missing phone verification — skip Kaggle API, use Drive method instead |
| Files missing after switching runtime | Local `/content/` storage was wiped — remount Drive and re-extract |
