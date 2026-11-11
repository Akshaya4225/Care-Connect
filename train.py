"""
Train a basic CNN for a folder-based image dataset.

Expected structure:
dataset/
  train/
    class_1/ image1.jpg ...
    class_2/ ...
  validation/
    class_1/ ...
    class_2/ ...

Each class folder name becomes the prediction label. Keep the same class folders
in train and validation. For a 22-class project, provide 22 class folders.
"""
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "validation"
MODEL_DIR = BASE_DIR / "models"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

def main():
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise SystemExit("Dataset folders not found. Read README.md for the required structure.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical", shuffle=True, seed=SEED
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical", shuffle=False
    )

    class_names = train_ds.class_names
    if len(class_names) < 2:
        raise SystemExit("At least two class folders are required.")
    if set(class_names) != set(val_ds.class_names):
        raise SystemExit("Train and validation folders must contain the same class names.")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
    ], name="augmentation")

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        augmentation,
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(192, 3, activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.35),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(len(class_names), activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    MODEL_DIR.mkdir(exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "skin_disease_model.keras",
            monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2)
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks
    )

    best_model = tf.keras.models.load_model(MODEL_DIR / "skin_disease_model.keras")
    loss, accuracy = best_model.evaluate(val_ds, verbose=0)
    (MODEL_DIR / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    report = {
        "validation_accuracy": round(float(accuracy), 4),
        "validation_loss": round(float(loss), 4),
        "classes": class_names,
        "epochs_completed": len(history.history["loss"])
    }
    (MODEL_DIR / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nTraining complete. Start the web app with: python app.py")

if __name__ == "__main__":
    main()
