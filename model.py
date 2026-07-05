import tensorflow as tf
from tensorflow.keras import layers, models


IMG_SIZE = (224, 224)


def build_hybrid_cnn(input_shape=(224, 224, 3), num_classes: int = 2) -> tf.keras.Model:
    """
    Build a hybrid CNN model for blood cancer image classification.

    Hybrid characteristics:
    - Standard convolution + max pooling in the first block
    - Multi-branch convolution block with different kernel sizes (Inception-style)
    - Depthwise separable convolution block for efficient feature extraction
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Block 1: Standard convolution
    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", name="conv_block1_conv")(inputs)
    x = layers.BatchNormalization(name="conv_block1_bn")(x)
    x = layers.MaxPool2D(pool_size=2, name="conv_block1_pool")(x)

    # Block 2: Multi-branch convolution (hybrid / Inception-style)
    branch1 = layers.Conv2D(64, kernel_size=1, padding="same", activation="relu", name="conv_block2_branch1")(x)
    branch2 = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu", name="conv_block2_branch2")(x)
    branch3 = layers.Conv2D(64, kernel_size=5, padding="same", activation="relu", name="conv_block2_branch3")(x)
    x = layers.Concatenate(name="conv_block2_concat")([branch1, branch2, branch3])
    x = layers.BatchNormalization(name="conv_block2_bn")(x)
    x = layers.MaxPool2D(pool_size=2, name="conv_block2_pool")(x)

    # Block 3: Depthwise separable convolution (another hybrid aspect)
    x = layers.SeparableConv2D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="last_conv_block",
    )(x)
    x = layers.BatchNormalization(name="conv_block3_bn")(x)
    x = layers.MaxPool2D(pool_size=2, name="conv_block3_pool")(x)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="hybrid_cnn_blood_cancer")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a trained model from disk.
    """
    return tf.keras.models.load_model(model_path)



