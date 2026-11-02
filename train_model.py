"""
Train a small CNN facial-expression classifier.

Expected dataset folder:
dataset/
  angry/
  disgust/
  fear/
  happy/
  neutral/
  sad/
  surprise/

Place labelled face images in the matching folders. Images are resized to
48x48 grayscale. After training, this script creates emotion_model.keras
and class_indices.json in the same folder as this script.
"""
import json
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 30
SEED = 42

if not DATA_DIR.exists():
    raise FileNotFoundError(
        "Create a 'dataset' folder and add one subfolder per emotion class. "
        "See README.txt for the required structure."
    )

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
(ROOT / "class_indices.json").write_text(
    json.dumps(class_names, indent=2), encoding="utf-8"
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.08),
])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(48, 48, 1)),
    tf.keras.layers.Rescaling(1.0 / 255.0),
    augmentation,
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.30),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.40),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        ROOT / "emotion_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

loss, accuracy = model.evaluate(val_ds, verbose=0)
print(f"Validation accuracy: {accuracy * 100:.2f}%")
print(f"Saved model: {ROOT / 'emotion_model.keras'}")
print(f"Saved class labels: {ROOT / 'class_indices.json'}")
