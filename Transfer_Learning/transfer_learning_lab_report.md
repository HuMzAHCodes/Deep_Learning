# Lab Report: Transfer Learning with VGG16 on the Dogs-vs-Cats Dataset

**Course:** Deep Learning / Generative AI
**Topic:** Transfer Learning — Feature Extraction and Fine-Tuning
**Environment:** Google Colab (T4 GPU), TensorFlow / Keras, Kaggle dataset `salader/dogsVScats`

---

## 1. Objective

To classify images of cats and dogs by reusing a convolutional neural network (VGG16) that was pre-trained on ImageNet, instead of training a network from scratch. The lab explores three progressively more powerful transfer-learning strategies and compares their behaviour:

1. Feature extraction **without** data augmentation
2. Feature extraction **with** data augmentation
3. **Fine-tuning** the top convolutional block

Each strategy shares the same backbone and dataset, so the differences in results isolate the effect of that one technique.

---

## 2. The underlying idea: what transfer learning is

A convolutional network trained on a large, general dataset (ImageNet: ~1.4M images, 1000 classes) learns a hierarchy of visual features — edges and colours in early layers, textures and simple shapes in the middle, and object-part detectors near the top. These low- and mid-level features are **generic**: an edge detector useful for recognising a truck is equally useful for recognising a dog.

Transfer learning reuses those learned features on a new, smaller task. Rather than learning "what an edge looks like" all over again from a few thousand images, we keep the pre-trained feature extractor and only teach the network the final decision ("is this a cat or a dog?"). This is faster, needs far less data, and usually generalises better than training from scratch.

There are two levers, and this lab uses both:

- **Feature extraction** — freeze the pre-trained layers, train only a new classifier on top.
- **Fine-tuning** — additionally unfreeze a few of the top pre-trained layers and let them adapt slightly to the new domain.

---

## 3. The convolutional base (the pre-trained feature extractor)

The backbone is VGG16 with its ImageNet weights, loaded **without** its original classifier.

```python
from keras.applications.vgg16 import VGG16

conv_base = VGG16(
    weights='imagenet',       # reuse features learned on ImageNet
    include_top=False,        # drop VGG16's original 1000-class dense head
    input_shape=(150, 150, 3) # our images are 150x150 RGB
)
```

Two arguments carry the whole concept:

- `weights='imagenet'` brings in the pre-trained filters — this is what we are transferring.
- `include_top=False` discards the original fully-connected classifier, because its 1000 ImageNet categories are irrelevant to a 2-class cat/dog problem. We keep only the convolutional feature extractor.

For a 150×150 input, VGG16's five pooling stages shrink the spatial size by a factor of 32, so the base outputs a feature map of shape **4 × 4 × 512**. This tensor is the "distilled" representation of the image that our own classifier will consume.

---

## 4. Feature extraction via freezing

The defining action of feature extraction is freezing the base so its weights are not updated during training.

```python
conv_base.trainable = False
```

With the base frozen, back-propagation only adjusts the new classifier we add on top. The pre-trained filters stay exactly as ImageNet left them. This matters for two reasons: it keeps training fast (only ~2 million parameters train instead of ~17 million), and it protects the valuable pre-trained features from being wrecked by large, noisy gradients coming from a freshly-initialised, untrained head.

If we skipped this step, the random head would push huge error signals back into the base during the first epochs and destroy the very features we wanted to reuse.

---

## 5. The custom classifier head

On top of the frozen base we attach a small trainable classifier suited to our task.

```python
from keras import Sequential
from keras.layers import Dense, Flatten

model = Sequential()
model.add(conv_base)                          # frozen feature extractor
model.add(Flatten())                          # 4x4x512  ->  8192-vector
model.add(Dense(256, activation='relu'))      # learnable hidden layer
model.add(Dense(1, activation='sigmoid'))     # binary output: cat vs dog
```

`Flatten` turns the 4×4×512 feature map into a single 8192-element vector. The `Dense(256, relu)` layer learns combinations of those features, and the final `Dense(1, sigmoid)` squashes the output to a probability between 0 and 1 — one output neuron is enough for binary classification, where values near 0 mean one class and near 1 the other.

---

## 6. The data pipeline and normalisation

Images are streamed from disk in labelled batches. Keras infers the class label from the sub-folder name (`cats/`, `dogs/`).

```python
train_ds = keras.utils.image_dataset_from_directory(
    directory=train_dir, labels='inferred', label_mode='int',
    batch_size=32, image_size=(150, 150)
)
validation_ds = keras.utils.image_dataset_from_directory(
    directory=test_dir, labels='inferred', label_mode='int',
    batch_size=32, image_size=(150, 150)
)
```

Raw pixel values range from 0–255. Neural networks train more stably on small, normalised inputs, so each image is rescaled to the 0–1 range:

```python
def process(image, label):
    image = tensorflow.cast(image / 255., tensorflow.float32)
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)
```

The same normalisation is applied to both training and validation data so the model sees inputs on a consistent scale.

---

## 7. Training configuration

The model is compiled with an optimiser, a loss function, and a metric.

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

`binary_crossentropy` is the correct loss for a single-sigmoid, two-class output: it penalises confident wrong predictions heavily and rewards confident correct ones. `adam` is a robust default optimiser for training the new head. `accuracy` is tracked as the human-readable performance metric.

Training then runs for a fixed number of epochs, with the validation set evaluated after each epoch:

```python
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
```

The returned `history` object stores per-epoch training and validation accuracy and loss, which we plot to diagnose learning behaviour.

---

## 8. Data augmentation (second strategy)

Feature extraction alone can overfit: with only the head learning and the same images shown every epoch, the model memorises the training set. Data augmentation combats this by showing slightly altered versions of each image every epoch — flipped, zoomed, rotated — so the model can never rely on memorising exact pixels and must learn more robust features.

The original notebook used the now-deprecated `ImageDataGenerator`. The modern equivalent uses augmentation **layers** applied inside the data pipeline:

```python
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomZoom(0.2),
    keras.layers.RandomRotation(0.1),
])

# Augment ONLY the training data, on the fly:
train_ds_aug = train_ds.map(
    lambda image, label: (data_augmentation(image, training=True), label)
)
```

A key design point: augmentation is applied **only to the training set, never to validation**. Validation must measure performance on clean, real images; augmenting it would give a misleading score.

Because augmentation lives in the data pipeline and not in the model, the same `model` object is reused and simply re-fit on the augmented stream — no architectural change is needed.

> **Note on the experiment design:** if the `model` is reused directly, it still holds the weights learned in the first run, so training continues from there rather than starting fresh. For a clean "with vs without augmentation" comparison, rebuild the model (re-run Sections 5 and 4) before fitting on the augmented data.

---

## 9. Fine-tuning (third strategy)

Fine-tuning goes one step further: after (or instead of) freezing, we unfreeze the **top** convolutional block and let it adapt to cats and dogs, while keeping the lower, more generic blocks frozen.

```python
conv_base.trainable = True

# Freeze everything BELOW block5; allow block5 onward to train.
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    layer.trainable = set_trainable
```

Only the highest block is unfrozen because the top layers encode the most **task-specific** features (object parts), which benefit most from adaptation, whereas the bottom layers encode universal features (edges) that are best left untouched.

Fine-tuning is then trained with a **very small learning rate**:

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
```

The tiny learning rate (`1e-5`) is essential. Large updates would overwrite the delicate pre-trained filters and erase the advantage of transfer learning; small updates only nudge them toward the new domain. (Note that `learning_rate=` replaces the old, removed `lr=` argument.)

**Best-practice ordering:** fine-tuning ideally follows a completed feature-extraction phase, so the classifier head is already sensible before the top conv block is unfrozen. Unfreezing while the head is still random risks damaging the base with large early gradients.

---

## 10. Reading the learning curves (evaluation)

Results are judged by plotting training vs validation accuracy and loss across epochs.

```python
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='blue', label='validation')
plt.legend(); plt.show()
```

Interpretation guide:

- **Both curves rising together, small gap** → healthy learning and good generalisation.
- **Training accuracy high, validation accuracy plateauing with a widening gap** → overfitting.
- **Validation loss turning upward while training loss keeps falling** → classic overfitting signal; the model is memorising.

This is the lens used to compare the three strategies in the next section.

---

## 11. Expected results and discussion

*(Record your own observed final train/validation accuracies from each run in the table; the qualitative pattern below is what these experiments typically show.)*

| Strategy | Trainable params | Speed | Typical behaviour |
|---|---|---|---|
| Feature extraction, no augmentation | ~2M (head only) | Fastest | High train accuracy, noticeable train/val gap (overfitting) |
| Feature extraction, with augmentation | ~2M (head only) | Slower per epoch | Smaller gap, validation tracks training more closely |
| Fine-tuning (block5 unfrozen) | ~9M | Slowest | Highest validation accuracy when trained carefully with low LR |

The trend illustrates the central lesson of the lab. Plain feature extraction is quick and already strong because ImageNet features transfer well to cats and dogs, but with a large head and unaltered images it tends to overfit. Data augmentation regularises the model by removing the option to memorise, narrowing the gap between training and validation performance at the cost of longer training. Fine-tuning extracts the last bit of performance by letting the top block specialise to the new domain, but only works when the learning rate is kept extremely small so the transferred features are preserved rather than destroyed.

---

## 12. Conclusion

Transfer learning lets a small dataset benefit from features learned on a massive one. Freezing a pre-trained base and training a lightweight head (feature extraction) gives a fast, strong baseline; augmenting the training images reduces overfitting; and carefully fine-tuning the top convolutional block with a tiny learning rate yields the best accuracy. The three strategies form a natural progression from cheapest-and-simplest to most-powerful, and the right choice depends on how much data, compute, and accuracy the task demands.

---

## Appendix A: Environment setup notes (data acquisition)

Getting the dataset into Colab required correcting two issues in the older code:

1. **Dataset slug is case-sensitive.** The correct Kaggle reference is `salader/dogsVScats`, not `salader/dogs-vs-cats`. Kaggle returns a misleading `403 Forbidden` on `GetDatasetMetadata` for an unresolved slug, which can be mistaken for an auth error. Verify with `!kaggle datasets list -s dogs-vs-cats` and copy the exact `ref`.
2. **Authentication uses the Legacy API Key.** Kaggle's newer single `KGAT_` token was unreliable across CLI/kagglehub versions; the classic `kaggle.json` (username + key) stored as Colab secrets `KAGGLE_USERNAME` / `KAGGLE_KEY` works reliably.

```python
!pip install -q --upgrade kaggle
import os
from google.colab import userdata
os.environ.pop("KAGGLE_API_TOKEN", None)                      # clear stale token
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME").strip()
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY").strip()
!kaggle datasets download -d salader/dogsVScats --unzip -p /content/dogs-vs-cats
```

The extracted structure is `train/{cats,dogs}` and `test/{cats,dogs}`, which `image_dataset_from_directory` reads directly.
