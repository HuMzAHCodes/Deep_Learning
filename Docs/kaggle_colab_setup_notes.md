# Kaggle → Colab Data Setup: Known Issues & Proven Fix

> Paste this file as context when asking an AI to modernize the data-loading /
> Kaggle-auth cells of an older transfer-learning notebook (Colab).
> It documents the exact traps we hit and the working solution, so you can
> skip the debugging and jump straight to the fix.

## Task context

The notebooks use the **`salader` dogs-vs-cats** dataset on Kaggle, pulled into
Google Colab. The original cells used the old pattern:

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets download -d salader/dogs-vs-cats
# + manual zipfile extraction
```

This no longer works cleanly on current Colab + current Kaggle. Two separate
things changed, and they masked each other.

## The single most important gotcha (the real root cause)

**The dataset slug in the old code is WRONG, and Kaggle returns a misleading
`403 Forbidden` instead of a `404 Not Found` for a slug it can't resolve.**

- Old/wrong slug: `salader/dogs-vs-cats`
- Correct slug (case-sensitive!): **`salader/dogsVScats`**

Because the error is a `403 ... DatasetApiService/GetDatasetMetadata`, it *looks*
exactly like an authentication/permission problem, so it's easy to burn an hour
on auth fixes that were never the issue.

**Always verify the slug first** before touching auth:

```python
!kaggle datasets list -s dogs-vs-cats
```

Read the `ref` column in the output and copy the exact slug (capitalization
included). For this dataset it is `salader/dogsVScats`.

## Secondary gotcha: Kaggle's new token format

Kaggle's "Create New Token" now yields a **single token starting with `KGAT_`**
(set via the `KAGGLE_API_TOKEN` env var / Colab secret), NOT the old
`kaggle.json` with `username` + `key`.

This new token was flaky across `kaggle` CLI / `kagglehub` versions in our
environment and kept failing. Symptoms while chasing it:

- Different kagglehub versions expect different Colab-secret names
  (`KAGGLE_API_TOKEN` vs the pair `KAGGLE_USERNAME` / `KAGGLE_KEY`).
- An outdated `kaggle` CLI (e.g. `2.0.2`) predates the `KGAT_` scheme.
- A stale `KAGGLE_API_TOKEN` env var set earlier in the session can **hijack**
  auth even after you switch to username/key, until you restart the runtime.

**Proven workaround: use the Legacy API Key instead of the KGAT token.**
On Kaggle: Settings → API → **"Create Legacy API Key"** (under *Legacy API
Credentials*) → downloads the classic `kaggle.json` containing `username` + `key`.
Every version of the CLI/kagglehub reads these two reliably.

## Diagnostic ladder (what a 403 actually means here)

Work through these in order; stop at the first that fails:

1. **Wrong slug** → most likely cause. Verify with `kaggle datasets list -s ...`.
   `403` on `GetDatasetMetadata` for a public dataset is usually this.
2. **Credentials not applied** → `403`/`401`. A stale `KAGGLE_API_TOKEN` env var
   overriding username/key; fix by `os.environ.pop("KAGGLE_API_TOKEN", None)` and
   restarting the runtime.
3. **Outdated CLI** → `!pip install -q --upgrade kaggle`.
4. **Account not phone-verified** → genuine `403` on downloads; verify phone in
   Kaggle Settings. (Was NOT our cause once the slug was fixed.)

Note: `403 Forbidden` = "credentials fine, not allowed / not found";
`401 Unauthorized` = "bad credentials". Distinguishing the two saves time.

## Proven working cells (current Colab, 2025+)

**Cell 1 — up-to-date CLI**
```python
!pip install -q --upgrade kaggle
```

**Cell 2 — Legacy credentials via Colab Secrets**
Store the `username` and `key` from the Legacy `kaggle.json` as two Colab secrets
(🔑 icon): `KAGGLE_USERNAME` and `KAGGLE_KEY` (Notebook access ON).
```python
import os
from google.colab import userdata

# Clear any stale KGAT token that could hijack auth.
os.environ.pop("KAGGLE_API_TOKEN", None)

# Legacy creds, whitespace-stripped.
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME").strip()
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY").strip()
```

**Cell 3 — download + auto-extract (note the correct slug)**
```python
# Correct, case-sensitive slug is salader/dogsVScats (NOT dogs-vs-cats).
# --unzip extracts in place; -p sets the destination folder.
!kaggle datasets download -d salader/dogsVScats --unzip -p /content/dogs-vs-cats
dataset_path = "/content/dogs-vs-cats"
print("Dataset ready at:", dataset_path)
```

**Cell 4 — locate the class folders**
```python
import os
print(os.listdir(dataset_path))               # ['test', 'train', 'catsvsdogs']
train_dir = os.path.join(dataset_path, "train")
test_dir  = os.path.join(dataset_path, "test")
print("Train classes:", os.listdir(train_dir))  # ['dogs', 'cats']
print("Test  classes:", os.listdir(test_dir))   # ['dogs', 'cats']
```

Confirmed folder layout after extraction:
```
/content/dogs-vs-cats/
├── train/   ->  cats/ , dogs/
├── test/    ->  cats/ , dogs/
└── catsvsdogs/   (extra, unused)
```

## Notes for whoever runs this next

- Switching runtime type (e.g. CPU → GPU) **restarts the session and wipes
  everything**, so you must Run-All from the top. All cells above are
  rerun-safe (re-download just overwrites; no errors).
- Delete any leftover broken download cell using the wrong slug so
  "Run all" doesn't hit it.
- For VGG16 training on ~20k images, use a **GPU runtime** (Runtime → Change
  runtime type → T4 GPU). Verify with:
  ```python
  import tensorflow as tf
  print("GPU:", tf.config.list_physical_devices('GPU'))
  ```
