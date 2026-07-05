import base64
from io import BytesIO
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from model import IMG_SIZE


def _make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = "last_conv_block",
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a single preprocessed image.

    Parameters
    ----------
    img_array : np.ndarray
        Array of shape (1, H, W, 3), normalized to [0, 1].
    model : tf.keras.Model
        Trained Keras model.
    last_conv_layer_name : str
        Name of the last convolutional layer used for Grad-CAM.
    """
    # Ensure batch dimension
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)

    # Get the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Predicted class index
        pred_index = tf.argmax(predictions[0])
        pred_output = predictions[:, pred_index]

    # Gradient of the top predicted class with respect to the outputs of the last conv layer
    grads = tape.gradient(pred_output, conv_outputs)

    # Vector of mean intensity of the gradients over each channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def _overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """
    Overlay Grad-CAM heatmap on top of the original image.
    """
    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap
    import matplotlib.cm as cm

    colormap = cm.get_cmap("jet")
    colored_heatmap = colormap(heatmap)
    colored_heatmap = np.uint8(255 * colored_heatmap[:, :, :3])

    # Resize to match original image
    heatmap_img = Image.fromarray(colored_heatmap).resize(original_image.size)

    # Overlay
    overlay = Image.blend(original_image.convert("RGB"), heatmap_img.convert("RGB"), alpha)
    return overlay


def generate_explanation_base64(
    model: tf.keras.Model,
    original_image: Image.Image,
    last_conv_layer_name: str = "last_conv_block",
) -> Tuple[np.ndarray, str]:
    """
    Generate prediction probabilities and a Grad-CAM explanation as base64 PNG.

    Returns
    -------
    probs : np.ndarray
        Model prediction probabilities for the input image.
    heatmap_b64 : str
        Base64-encoded PNG image of the Grad-CAM overlay.
    """
    # Preprocess image
    img_resized = original_image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predictions
    probs = model.predict(img_array)[0]

    # Grad-CAM heatmap
    heatmap = _make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    overlay_img = _overlay_heatmap_on_image(original_image, heatmap)

    # Encode overlay as base64
    buffer = BytesIO()
    overlay_img.save(buffer, format="PNG")
    buffer.seek(0)
    heatmap_b64 = base64.b64encode(buffer.read()).decode("utf-8")

    return probs, heatmap_b64



