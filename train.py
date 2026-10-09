"""
Smart Waste Classification Using a Deep Neural Network
Dataset folders expected:
dataset/
  Paper/
  Plastic/
  Metal/
  Organic/

This script splits images into train/validation/test sets, trains a DNN,
evaluates it, and saves the model and plots.
"""
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
IMG_SIZE = (128, 128)
BATCH_SIZE = 4
EPOCHS = 30
CATEGORIES = ["Metal", "Organic", "Paper", "Plastic"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path("dataset")
SPLIT_DIR = Path("waste_split")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Find category folders, even if the dataset has an extra parent directory.
category_paths = {}
for category in CATEGORIES:
    matches = [p for p in ROOT.rglob("*")
               if p.is_dir() and p.name.lower() == category.lower()]
    if not matches:
        raise FileNotFoundError(
            f"Could not find the '{category}' folder inside {ROOT.resolve()}"
        )
    category_paths[category] = matches[0]

# Re-create the split folders on each run.
if SPLIT_DIR.exists():
    shutil.rmtree(SPLIT_DIR)
for split in ["train", "validation", "test"]:
    for category in CATEGORIES:
        (SPLIT_DIR / split / category).mkdir(parents=True, exist_ok=True)

for category, folder in category_paths.items():
    images = [p for p in folder.rglob("*")
              if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    random.shuffle(images)
    n = len(images)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)
    groups = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:],
    }
    for split, files_list in groups.items():
        for i, image_path in enumerate(files_list):
            destination = SPLIT_DIR / split / category / f"{category}_{i}{image_path.suffix.lower()}"
            shutil.copy2(image_path, destination)

print("Dataset split counts:")
for split in ["train", "validation", "test"]:
    print(f"\n{split.upper()}")
    for category in CATEGORIES:
        count = len(list((SPLIT_DIR / split / category).glob("*")))
        print(f"{category}: {count}")

def load_split(split):
    return keras.utils.image_dataset_from_directory(
        SPLIT_DIR / split,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        class_names=CATEGORIES,
        shuffle=(split == "train"),
        seed=SEED,
    )

train_ds = load_split("train").prefetch(tf.data.AUTOTUNE)
val_ds = load_split("validation").prefetch(tf.data.AUTOTUNE)
test_ds = load_split("test").prefetch(tf.data.AUTOTUNE)

model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1.0 / 255),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(4, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping],
)

test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

# Accuracy graph
plt.figure(figsize=(7, 4))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy.png", dpi=150)
plt.close()

# Loss graph
plt.figure(figsize=(7, 4))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "loss.png", dpi=150)
plt.close()

# Confusion matrix and report
y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
probabilities = model.predict(test_ds, verbose=0)
y_pred = np.argmax(probabilities, axis=1)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CATEGORIES)
disp.plot(cmap="Blues", values_format="d")
plt.title("Waste Classification Confusion Matrix")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
plt.close()

print("\nClassification report:")
print(classification_report(
    y_true, y_pred, labels=[0, 1, 2, 3],
    target_names=CATEGORIES, zero_division=0
))

model.save("waste_classifier.keras")
print("\nSaved model to waste_classifier.keras")
print(f"Saved graphs in {RESULTS_DIR.resolve()}")
