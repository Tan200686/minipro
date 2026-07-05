import json
from pathlib import Path

import tensorflow as tf

from model import build_hybrid_cnn, IMG_SIZE


DATA_DIR = Path("data")
MODEL_PATH = Path("hybrid_cnn_blood_cancer.h5")
CLASS_NAMES_PATH = Path("class_names.txt")

BATCH_SIZE = 32
EPOCHS = 20
SEED = 123


def get_datasets():
    """
    Create training and validation datasets from a directory structure:

    data/
      train/
        class1/
        class2/
      val/
        class1/
        class2/
    """
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory '{train_dir}' not found. "
            "Please create it and organize images in class subfolders."
        )
    if not val_dir.exists():
        raise FileNotFoundError(
            f"Validation directory '{val_dir}' not found. "
            "Please create it and organize images in class subfolders."
        )

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    # Save class names before applying prefetch (prefetch may drop attributes)
    class_names = train_ds.class_names

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def main():
    train_ds, val_ds, class_names = get_datasets()
    num_classes = len(class_names)
    print(f"Detected classes: {class_names}")

    # Build model
    model = build_hybrid_cnn(input_shape=IMG_SIZE + (3,), num_classes=num_classes)
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save final model (in case best checkpoint is not the last)
    model.save(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}")

    # Save class names
    with CLASS_NAMES_PATH.open("w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"Class names saved to {CLASS_NAMES_PATH}")

    # Optionally, save training history
    history_path = Path("training_history.json")
    with history_path.open("w", encoding="utf-8") as f:
        json.dump({k: [float(vv) for vv in v] for k, v in history.history.items()}, f)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()



