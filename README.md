# Food101 Classification with TensorFlow

This project involves the classification of 101 food items using TensorFlow. The dataset is loaded from TensorFlow Datasets (TFDS) and the model is built using TensorFlow and Keras, specifically leveraging the EfficientNetB7 architecture. The project includes data preprocessing, model training, and evaluation with various enhancements for better performance.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/meetvyas3012/Food-Classifier.git
cd Food-Classifier
pip install -r requirements.txt
```

## Dataset

The dataset used for this project is the Food101 dataset, which is available through TensorFlow Datasets. To list all available datasets:

```python
import tensorflow_datasets as tfds
tfds.list_builders()
```

## Importing the Dataset

To load and split the dataset:

```python
import tensorflow as tf
import tensorflow_datasets as tfds

(train_data, test_data), ds_info = tfds.load(
    name="food101",
    split=["train", "validation"],
    shuffle_files=True,
    as_supervised=True,  # downloads file in tuple format
    with_info=True
)
```

## Preprocessing

Define a function to resize and preprocess images:

```python
def preprocess_img(img, label):
    img = tf.image.resize(img, [224, 224])
    return tf.cast(img, tf.float32), label
```

Apply the preprocessing to the dataset:

```python
train_data = train_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)
```

## Model

Create the base model using EfficientNetB7:

```python
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy(policy="mixed_float16")

base_model = tf.keras.applications.EfficientNetB7(include_top=False)
base_model.trainable = False

input_layer = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
x = base_model(input_layer, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name="pooling_layer")(x)
x = tf.keras.layers.Dense(len(ds_info.features["label"].names))(x)
output_layer = tf.keras.layers.Activation("softmax", dtype=tf.float32, name="softmax_32")(x)

model = tf.keras.Model(input_layer, output_layer)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="Adam",
    metrics=["accuracy"]
)
```

## Training

Train the model with initial parameters:

```python
history_1 = model.fit(
    train_data,
    epochs=3,
    steps_per_epoch=len(train_data),
    validation_data=test_data,
    validation_steps=int(0.15 * len(test_data)),
    callbacks=[create_tensorboard_callback("training_logs", "efficientnetb0_101_classes_all_data_feature_extract"),
               model_checkpoint]
)
```

Fine-tune the model:

```python
for layer in model.layers:
    layer.trainable = True

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy"]
)

history_2 = model.fit(
    train_data,
    epochs=100,
    validation_data=test_data,
    steps_per_epoch=len(train_data),
    validation_steps=int(len(test_data)),
    callbacks=[create_tensorboard_callback("training_logs", "fine_tuned"),
               early_stopping_callback,
               reduce_lr]
)
```

## Evaluation

Evaluate the model on the test data:

```python
model.evaluate(test_data)
```

## Results

Plot the training history:

```python
import pandas as pd

pd.DataFrame(history_2.history).plot()
```


## Acknowledgements

- TensorFlow and TensorFlow Datasets teams for providing the tools and datasets.
- EfficientNetB7 model authors for their contributions to model architecture.

---

This README provides a comprehensive guide to the project, covering all key aspects from installation to evaluation. Adjust any URLs, repository names, or other specifics to fit your actual project details.
