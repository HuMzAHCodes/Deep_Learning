# Lab Report: Multi-Output Age & Gender Prediction with ResNet50 (UTKFace)

**Course:** Deep Learning / Generative AI
**Topic:** Transfer Learning + Multi-Task Learning (one input → two predictions)
**Environment:** Google Colab (T4 GPU), TensorFlow 2.20 / Keras 3, Kaggle dataset `jangedoo/utkface-new`

---

## 1. Objective

To build a single neural network that takes one face image and simultaneously predicts **two different things**:

- **Age** — a continuous number in years (a *regression* task)
- **Gender** — male or female (a *binary classification* task)

The network reuses a ResNet50 backbone pre-trained on ImageNet (transfer learning) as a shared feature extractor, then splits into two task-specific "heads." This lab therefore combines two ideas: **transfer learning** (reusing pre-trained features) and **multi-task learning** (predicting several outputs from one shared representation).

---

## 2. The core idea: multi-task (multi-output) learning

Most models produce one output. Here, a single model produces two, because age and gender can be inferred from the *same* visual features of a face. Instead of training two separate networks, we train one network with:

- a **shared body** (ResNet50) that extracts general face features once, and
- **two separate output branches**, each specialising in one task.

The advantages are efficiency (features are computed once, not twice) and shared learning (the backbone learns representations useful to both tasks). The challenge, addressed in Section 9, is that the two tasks have very different loss scales and must be balanced.

Because the model has a branching, multi-output shape, it cannot be built with the simple `Sequential` API — it requires the **Functional API**, where layers are called on tensors and outputs are wired explicitly.

---

## 3. The dataset: UTKFace and label-in-filename encoding

UTKFace is a collection of cropped, aligned face images. Crucially, **the labels are stored inside each filename**, not in a separate CSV. Each file is named:

```
age_gender_race_datetime.jpg      e.g.  25_0_0_20170116.jpg
```

So the label for each image is parsed by splitting the filename on underscores:

- `split('_')[0]` → **age** (integer years)
- `split('_')[1]` → **gender** (0 = male, 1 = female)

```python
age, gender, img_path = [], [], []
for file in os.listdir(folder_path):
    age.append(int(file.split('_')[0]))
    gender.append(int(file.split('_')[1]))
    img_path.append(file)
```

These lists are collected into a pandas DataFrame that pairs every image filename with its two labels:

```python
df = pd.DataFrame({'age': age, 'gender': gender, 'img': img_path})
```

---

## 4. Train / test split

The data is shuffled once with a fixed seed (for reproducibility), then partitioned by row position: the first 20,000 images form the training set and the remainder form the test set.

```python
train_df = df.sample(frac=1, random_state=0).iloc[:20000]
test_df  = df.sample(frac=1, random_state=0).iloc[20000:]
```

Using the same `random_state` for both ensures the shuffle order is identical, so the two slices are disjoint (no image appears in both sets).

---

## 5. The input pipeline (tf.data)

The images must be read from disk, decoded, resized to a fixed size, and scaled before entering the network. The pipeline is built with `tf.data`, which decodes images **in parallel** and **prefetches** the next batch while the GPU is busy — keeping the GPU fed rather than idle.

```python
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(dframe, training):
    paths   = folder_path + '/' + dframe['img'].values
    ages    = dframe['age'].values.astype('float32')
    genders = dframe['gender'].values.astype('float32')

    ds = tf.data.Dataset.from_tensor_slices((paths, ages, genders))

    def load(path, age, gender):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [200, 200]) / 255.0   # fixed size + [0,1] scaling
        return img, {'age': age, 'gender': gender}        # dict keys = output layer names

    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(1000)
        ds = ds.map(lambda img, y: (tf.image.random_flip_left_right(img), y),
                    num_parallel_calls=AUTOTUNE)           # light augmentation, train only
    return ds.batch(32).prefetch(AUTOTUNE)

train_ds = make_dataset(train_df, training=True)
test_ds  = make_dataset(test_df,  training=False)
```

Two design points are essential:

1. **Labels are returned as a dictionary** `{'age': ..., 'gender': ...}` whose keys match the names of the model's two output layers. This is how Keras routes each label to the correct head.
2. **Augmentation is applied to the training set only.** The test set is left as clean, real images so its score reflects true performance.

*(This pipeline replaced the original `ImageDataGenerator`; see Appendix A for why.)*

---

## 6. The shared backbone: frozen ResNet50 (transfer learning)

ResNet50, pre-trained on ImageNet, serves as the feature extractor. Its classifier top is removed and its weights are frozen, so it acts as a fixed function that turns a 200×200×3 image into a rich feature map.

```python
resnet = ResNet50(include_top=False, input_shape=(200, 200, 3))
resnet.trainable = False        # freeze: only the new heads will learn
```

`include_top=False` drops ResNet50's original 1000-class ImageNet classifier (irrelevant here), and `trainable = False` freezes the transferred features so they are reused rather than relearned. For a 200×200 input, the backbone outputs a feature map of shape **7 × 7 × 2048**.

---

## 7. The two-branch head (Functional API)

The frozen features are flattened and then fed into **two independent branches**, one per task. Each branch is a small stack of dense layers ending in a task-appropriate output neuron.

```python
output  = resnet.layers[-1].output
flatten = Flatten()(output)                 # 7x7x2048 -> single feature vector

# --- Age branch (regression) ---
dense1  = Dense(512, activation='relu')(flatten)
dense3  = Dense(512, activation='relu')(dense1)
output1 = Dense(1, activation='linear', name='age')(dense3)

# --- Gender branch (classification) ---
dense2  = Dense(512, activation='relu')(flatten)
dense4  = Dense(512, activation='relu')(dense2)
output2 = Dense(1, activation='sigmoid', name='gender')(dense4)

model = Model(inputs=resnet.input, outputs=[output1, output2])
```

Both branches start from the **same** `flatten` tensor (shared features) but then diverge into separate weights, so each can learn what its own task needs. The two branches are joined into one model by listing both outputs in `Model(outputs=[output1, output2])`.

---

## 8. Two output types: regression vs classification

The two heads differ precisely because the two tasks are mathematically different:

| | Age head | Gender head |
|---|---|---|
| Task type | Regression (continuous) | Binary classification |
| Final activation | `linear` (no squashing) | `sigmoid` (0–1 probability) |
| Loss | `mae` (mean absolute error) | `binary_crossentropy` |
| Output meaning | predicted age in years | P(gender = 1) |

The age head uses a linear output because age is an unbounded real number; squashing it would cap the range. The gender head uses sigmoid so its single output is a probability, thresholded at 0.5 to decide male vs female.

---

## 9. Balancing the two losses (loss weighting)

The final loss the model minimises is a *sum* of the two heads' losses. But those losses live on wildly different scales: age MAE is measured in years (early values around 16), while gender cross-entropy is a small number (around 0.6–1.3). If simply added, the large age loss dominates and the model barely bothers to learn gender.

The fix is `loss_weights`, which multiplies each task's loss before summing:

```python
model.compile(
    optimizer='adam',
    loss={'age': 'mae', 'gender': 'binary_crossentropy'},
    metrics={'age': 'mae', 'gender': 'accuracy'},
    loss_weights={'age': 1, 'gender': 99}
)
```

Weighting gender by 99 scales its small loss up to be comparable with (indeed, dominant over) the age loss, so gradients actually push the gender head to improve. Per-task `loss` and `metrics` dictionaries tell Keras which loss and which readable metric belong to each named output.

---

## 10. Training

Training runs on the two datasets, evaluating on the test set each epoch. Because the pipeline yields label dictionaries keyed by output name, Keras automatically sends age labels to the age head and gender labels to the gender head.

```python
history = model.fit(train_ds, epochs=10, validation_data=test_ds)
```

The training log shows a combined `loss` plus per-head readouts (`age_mae`, `gender_accuracy`), letting each task be monitored separately.

---

## 11. Inference on a new image

To predict on an uploaded photo, the image must be preprocessed **exactly** as in training — decoded, resized to 200×200, and scaled to [0, 1] — otherwise predictions are unreliable.

```python
raw = tf.io.read_file(img_name)
img = tf.io.decode_jpeg(raw, channels=3)
img = tf.image.resize(img, [200, 200]) / 255.0
img_batch = tf.expand_dims(img, axis=0)          # (1, 200, 200, 3)

age_pred, gender_pred = model.predict(img_batch, verbose=0)
predicted_age    = float(age_pred[0][0])
gender_prob      = float(gender_pred[0][0])       # 0..1
predicted_gender = "Female" if gender_prob > 0.5 else "Male"   # UTKFace: 0=Male, 1=Female
```

The model returns a list `[age_output, gender_output]`; the age value is read directly, and the gender probability is thresholded at 0.5. Supported input formats after decoding include JPEG, PNG, and BMP; HEIC/WEBP must be converted first.

---

## 12. Results and discussion

*(Fill in your observed final values from the training log.)*

| Output | Metric | Early epoch (observed) | Final epoch (record yours) |
|---|---|---|---|
| Gender | accuracy | ~0.51 (chance) | ___ |
| Age | MAE (years) | ~16 | ___ |

Expected behaviour and interpretation:

- **Gender** typically becomes the stronger head. It is an easy binary task, it was up-weighted, and starts near 0.5 (random guessing) before climbing.
- **Age is intrinsically harder** and, as a regression task on a frozen backbone trained for only 10 epochs, tends to remain rough — being off by several years is normal. The `age_mae` value is the honest yardstick: predictions are roughly that many years off on average.
- **Architecture note (improvement):** flattening a 7×7×2048 feature map produces a ~100k-length vector, so the first `Dense(512)` in each branch has ~50M parameters — the heads dominate the parameter count. Replacing `Flatten()` with `GlobalAveragePooling2D()` would shrink the heads dramatically, train faster, and usually generalise better. Unfreezing the top ResNet block (fine-tuning) with a very small learning rate would likely improve age accuracy further.

---

## 13. Conclusion

A single ResNet50-based network can predict age and gender together by sharing a frozen pre-trained backbone and branching into two task-specific heads — one regression (linear + MAE) and one classification (sigmoid + cross-entropy). The key practical lesson is **loss balancing**: when a model optimises multiple objectives with different scales, their losses must be weighted so no task is ignored. The result is an efficient multi-task model that reuses transferred features for both predictions at once.

---

## Appendix A: Environment issues encountered (and fixes)

Getting this notebook to run on current Colab required resolving three separate problems:

1. **Kaggle authentication.** The old `kaggle.json` upload pattern was replaced with the **Legacy API Key** stored as Colab Secrets (`KAGGLE_USERNAME` / `KAGGLE_KEY`). Kaggle's newer single `KGAT_` token was unreliable across CLI/library versions. The same account key works across notebooks — no need to regenerate per dataset.

2. **Deprecated data API.** The original code used `ImageDataGenerator` + `flow_from_dataframe(class_mode='multi_output')`, which were **removed in Keras 3** (the current Colab default). The clean fix was to rewrite the input pipeline with `tf.data` (Section 5) rather than pin an old Keras.

3. **GPU not actually attached / slow training.** An early run took ~50 minutes per epoch. `tf.config.list_physical_devices('GPU')` returned an empty list — the runtime was on CPU. Two contributing causes: the runtime type had to be set to **T4 GPU** (Runtime → Change runtime type), and forcing legacy Keras via `tf_keras` had pulled a **CPU-only TensorFlow build**. Dropping the legacy pin (staying on TF 2.20 / Keras 3) restored GPU support, and the `tf.data` pipeline removed the single-threaded data bottleneck. Epoch time fell from ~50 minutes to a few minutes.

```python
# Confirm GPU before training:
import tensorflow as tf
print(tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))   # must list a device
```
