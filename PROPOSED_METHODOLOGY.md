# III. PROPOSED METHODOLOGY

The system is built on a **backend-only architecture** using **Python, TensorFlow/Keras, and FastAPI**. It employs a **Hybrid Convolutional Neural Network (CNN)** architecture combined with **Explainable AI (XAI)** to maximize both diagnostic accuracy and interpretability for medical professionals.

**Fig. 1: Full System Flow Chart.**

```
┌─────────────────────────────────────────────────────────────┐
│                    Blood Smear Image Input                     │
│              (Microscopic Image: .jpg, .png, .bmp)             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Module 1: Image Preprocessing                       │
│  • Resize to 224×224 pixels                                  │
│  • RGB normalization                                          │
│  • Pixel value scaling [0, 1]                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│        Module 2: Hybrid CNN Feature Extraction                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Block 1: Standard Convolution                         │   │
│  │  • Conv2D(32, 3×3) + BatchNorm + MaxPool              │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Block 2: Multi-Branch Convolution (Inception-style)  │   │
│  │  • Branch 1: 1×1 Conv (64 filters)                  │   │
│  │  • Branch 2: 3×3 Conv (64 filters)                    │   │
│  │  • Branch 3: 5×5 Conv (64 filters)                    │   │
│  │  • Concatenate + BatchNorm + MaxPool                  │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Block 3: Depthwise Separable Convolution              │   │
│  │  • SeparableConv2D(128, 3×3) + BatchNorm + MaxPool   │   │
│  └──────────────────────────────────────────────────────┘   │
│                         │                                     │
│                         ▼                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Classification Head                                    │   │
│  │  • GlobalAveragePooling2D                             │   │
│  │  • Dropout(0.5)                                       │   │
│  │  • Dense(num_classes, softmax)                         │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│  Prediction      │          │  Module 3: XAI   │
│  Probabilities   │          │  (Grad-CAM)      │
│  [Normal, Cancer]│          │  Heatmap Gen.    │
└──────────────────┘          └──────────────────┘
         │                               │
         └───────────────┬───────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Module 4: API Response Generation                 │
│  • JSON: {predictions: [...], explanation: {base64: ...}}    │
└─────────────────────────────────────────────────────────────┘
```

## Module 1: Image Preprocessing Pipeline

To process blood smear images, we implement a standardized preprocessing pipeline optimized for deep learning inference:

**Resizing**: Input images are resized to a fixed resolution of 224×224 pixels using bilinear interpolation. This standardization ensures consistent input dimensions regardless of the original microscope magnification or image capture resolution.

**Color Space Normalization**: Images are converted to RGB color space (3 channels) to ensure compatibility with the CNN architecture. Grayscale images are duplicated across all three channels, while multi-channel images are preserved as-is.

**Pixel Value Scaling**: Raw pixel intensities (typically in the range [0, 255]) are normalized to the floating-point range [0, 1] by dividing by 255.0. This normalization accelerates gradient-based optimization during training and improves numerical stability during inference.

**Batch Dimension Addition**: A batch dimension is prepended to the image array, transforming the shape from (224, 224, 3) to (1, 224, 224, 3) to match the expected input format of the Keras model.

## Module 2: Hybrid CNN Architecture

Building on the work of [1] and [2], the system employs a **Hybrid Convolutional Neural Network** that combines multiple architectural paradigms to capture multi-scale features from blood cell morphology.

**Block 1: Standard Convolution Block**: The first block applies traditional 2D convolution with 32 filters of size 3×3, followed by batch normalization to stabilize training and max pooling (2×2) to reduce spatial dimensions. This block extracts low-level features such as edges, textures, and basic cellular structures.

**Block 2: Multi-Branch Convolution Block (Inception-style)**: Inspired by the Inception architecture [3], this block employs three parallel convolutional branches with different kernel sizes (1×1, 3×3, and 5×5), each producing 64 feature maps. The 1×1 branch captures point-wise features and reduces dimensionality, the 3×3 branch detects medium-scale patterns, and the 5×5 branch identifies larger morphological structures. The outputs are concatenated along the channel dimension, resulting in 192 feature maps (64×3), followed by batch normalization and max pooling. This hybrid approach enables the model to simultaneously recognize fine-grained cellular details (e.g., nucleus shape) and broader tissue-level patterns (e.g., cell density distribution).

**Block 3: Depthwise Separable Convolution Block**: To enhance computational efficiency while maintaining representational power, the third block utilizes depthwise separable convolution [4]. This operation factorizes standard convolution into two steps: (1) depthwise convolution, which applies a single filter per input channel, and (2) pointwise convolution, which combines the depthwise outputs using 1×1 filters. This block produces 128 feature maps with significantly fewer parameters than standard convolution, enabling deeper networks without excessive memory overhead.

**Classification Head**: The feature maps are aggregated using global average pooling, which reduces spatial dimensions to a single vector per channel. A dropout layer (rate=0.5) is applied to prevent overfitting, followed by a fully connected layer with softmax activation to produce probability distributions over cancer classes.

## Module 3: Explainable AI (Grad-CAM)

To address the "black box" nature of deep learning models in medical diagnostics, we implement **Gradient-weighted Class Activation Mapping (Grad-CAM)** [5], which generates visual explanations highlighting the image regions most influential in the model's prediction.

**Gradient Computation**: Upon receiving a preprocessed image, the system constructs a computational graph that tracks gradients of the predicted class score with respect to the activations of the last convolutional layer (`last_conv_block`). Using TensorFlow's automatic differentiation (GradientTape), the gradients are computed in a single forward-backward pass.

**Heatmap Generation**: The gradients are globally average-pooled across spatial dimensions to produce a weight vector, where each element corresponds to the importance of a feature channel. The heatmap is computed as a weighted sum of the convolutional feature maps, followed by ReLU activation to retain only positive contributions. The resulting heatmap is normalized to the range [0, 1] for visualization.

**Visual Overlay**: The normalized heatmap is resized to match the original image dimensions and converted to a color-coded visualization using the "jet" colormap (blue for low activation, red for high activation). The colored heatmap is blended with the original image at a transparency level (alpha=0.4) to create an overlay that clearly indicates which cellular regions the model identified as diagnostic indicators of cancer.

**Base64 Encoding**: The final overlay image is encoded as a base64 PNG string, enabling efficient transmission over HTTP APIs without requiring separate file storage or additional network requests.

## Module 4: RESTful API Integration

The backend exposes a **FastAPI-based RESTful API** that serves model predictions and explanations to client applications (web frontends, mobile apps, or medical imaging systems).

**Endpoint 1: `/predict`**: Accepts an image file via multipart form-data, performs preprocessing and inference using the hybrid CNN, and returns a JSON response containing class probabilities sorted in descending order. This lightweight endpoint is optimized for scenarios where only classification results are required.

**Endpoint 2: `/predict-with-explanation`**: Extends the prediction endpoint by additionally generating a Grad-CAM heatmap overlay. The response includes both the probability distribution and a base64-encoded PNG image of the explanation visualization. This endpoint is designed for clinical use cases where interpretability is critical for physician trust and diagnostic validation.

**Model Loading Strategy**: The trained model (`hybrid_cnn_blood_cancer.h5`) and class names (`class_names.txt`) are loaded into memory at application startup, ensuring sub-second inference latency. The model remains in memory for the lifetime of the server process, eliminating disk I/O overhead during prediction requests.

## Mathematical Formulation of Hybrid CNN

The hybrid CNN architecture operates on a hierarchical feature extraction framework, where each block transforms the input through a series of learnable operations.

**Standard Convolution Operation (Block 1)**:
For an input feature map **X** of shape (H, W, C_in), the convolution operation produces output **Y**:

**Y** = ReLU(BN(Conv2D(**X**, **W₁**, **b₁**)))

where **W₁** ∈ ℝ^(3×3×C_in×32) and **b₁** ∈ ℝ^32 are learnable weights and biases, BN denotes batch normalization, and ReLU is the rectified linear unit activation function.

**Multi-Branch Convolution (Block 2)**:
The Inception-style block computes three parallel branches:

**Y₁** = ReLU(Conv2D(**X**, **W₁×₁**, **b₁×₁**))  // 1×1 branch
**Y₂** = ReLU(Conv2D(**X**, **W₃×₃**, **b₃×₃**))  // 3×3 branch
**Y₃** = ReLU(Conv2D(**X**, **W₅×₅**, **b₅×₅**))  // 5×5 branch

The outputs are concatenated along the channel dimension:

**Y_concat** = Concat([**Y₁**, **Y₂**, **Y₃**])  // Shape: (H', W', 192)

**Depthwise Separable Convolution (Block 3)**:
This operation factorizes standard convolution into two steps:

**Z_depthwise** = DepthwiseConv2D(**X**, **W_dw**, **b_dw**)
**Z_pointwise** = Conv2D(**Z_depthwise**, **W_1×1**, **b_1×1**)

where the depthwise convolution applies a single filter per input channel, and the pointwise convolution combines channels using 1×1 kernels.

**Classification Output**:
The final prediction probabilities are computed as:

**p** = Softmax(**W_fc** · GAP(**Z_pointwise**) + **b_fc**)

where GAP denotes global average pooling, **W_fc** and **b_fc** are the fully connected layer parameters, and Softmax ensures that Σᵢ pᵢ = 1.

**Loss Function**: During training, the model minimizes the **sparse categorical cross-entropy** loss between the predicted probability distribution **p** and the true class label **y**:

L = -log(p_y)

where p_y is the predicted probability for the true class. This loss function is well-suited for multi-class classification tasks and provides stable gradients for optimization.

## Mathematical Formulation of Grad-CAM

The Grad-CAM algorithm generates a heatmap **H** that visualizes the importance of each spatial location in the input image for the predicted class.

**Gradient Computation**:
For a given input image **I** and predicted class score **S^c** (where c is the class index), the gradient of **S^c** with respect to the feature maps **A^k** of the last convolutional layer is computed:

∂S^c/∂A^k_ij

where k indexes the feature channels and (i, j) indexes spatial locations.

**Channel Weights**:
The importance weight α^k_c for each channel k is computed as the global average of the gradients:

α^k_c = (1/Z) Σᵢ Σⱼ (∂S^c/∂A^k_ij)

where Z = H × W is the normalization factor (spatial dimensions of the feature maps).

**Heatmap Generation**:
The heatmap **H** is computed as a weighted linear combination of the feature maps, followed by ReLU activation:

**H** = ReLU(Σₖ α^k_c · **A^k**)

The ReLU ensures that only features with positive influence on the class prediction are visualized.

**Normalization**:
The heatmap is normalized to the range [0, 1] for visualization:

**H_normalized** = **H** / (max(**H**) + ε)

where ε = 1e-8 is a small constant to prevent division by zero.

## Algorithms for Key Operations

To ensure robust and interpretable predictions, we implemented specific algorithms for image preprocessing, model inference, and explanation generation.

**Algorithm 1: Image Preprocessing Pipeline**

```
Input: Raw Image (I_raw)
Output: Preprocessed Tensor (I_processed)
1:  I_rgb ← ConvertToRGB(I_raw)
2:  I_resized ← Resize(I_rgb, (224, 224))
3:  I_array ← ConvertToNumpy(I_resized)
4:  I_normalized ← I_array / 255.0  // Scale to [0, 1]
5:  I_batched ← ExpandDims(I_normalized, axis=0)  // Shape: (1, 224, 224, 3)
6:  Return I_batched
```

**Algorithm 2: Hybrid CNN Inference**

```
Input: Preprocessed Image (I_batched), Trained Model (M)
Output: Class Probabilities (P)
1:  // Block 1: Standard Convolution
2:  F1 ← Conv2D(I_batched, filters=32, kernel=3×3)
3:  F1 ← BatchNormalization(F1)
4:  F1 ← ReLU(F1)
5:  F1 ← MaxPooling2D(F1, pool_size=2×2)
6:  
7:  // Block 2: Multi-Branch Convolution
8:  B1 ← Conv2D(F1, filters=64, kernel=1×1)
9:  B2 ← Conv2D(F1, filters=64, kernel=3×3)
10: B3 ← Conv2D(F1, filters=64, kernel=5×5)
11: F2 ← Concatenate([B1, B2, B3])
12: F2 ← BatchNormalization(F2)
13: F2 ← MaxPooling2D(F2, pool_size=2×2)
14: 
15: // Block 3: Depthwise Separable Convolution
16: F3 ← SeparableConv2D(F2, filters=128, kernel=3×3)
17: F3 ← BatchNormalization(F3)
18: F3 ← MaxPooling2D(F3, pool_size=2×2)
19: 
20: // Classification Head
21: F_global ← GlobalAveragePooling2D(F3)
22: F_dropout ← Dropout(F_global, rate=0.5)
23: P ← Dense(F_dropout, units=num_classes, activation=softmax)
24: 
25: Return P
```

**Algorithm 3: Grad-CAM Heatmap Generation**

```
Input: Preprocessed Image (I_batched), Trained Model (M), Class Index (c)
Output: Heatmap Overlay Image (H_overlay)
1:  // Step 1: Extract last convolutional layer
2:  LastConvLayer ← M.get_layer("last_conv_block")
3:  
4:  // Step 2: Create gradient computation model
5:  GradModel ← Model([M.input], [LastConvLayer.output, M.output])
6:  
7:  // Step 3: Compute gradients
8:  With GradientTape() as tape:
9:    ConvOutputs, Predictions ← GradModel(I_batched)
10:   PredScore ← Predictions[0, c]  // Score for class c
11: 
12: // Step 4: Compute gradients
13: Grads ← tape.gradient(PredScore, ConvOutputs)
14: 
15: // Step 5: Global average pooling of gradients
16: Alpha ← Mean(Grads, axis=(0, 1, 2))  // Shape: (num_channels,)
17: 
18: // Step 6: Weighted combination of feature maps
19: Heatmap ← Sum(Alpha[k] * ConvOutputs[0, :, :, k] for k in channels)
20: Heatmap ← ReLU(Heatmap)  // Retain only positive contributions
21: 
22: // Step 7: Normalize heatmap
23: Heatmap ← Heatmap / (Max(Heatmap) + 1e-8)
24: 
25: // Step 8: Resize and overlay on original image
26: HeatmapResized ← Resize(Heatmap, OriginalImage.size)
27: HeatmapColored ← ApplyColormap(HeatmapResized, "jet")
28: H_overlay ← Blend(OriginalImage, HeatmapColored, alpha=0.4)
29: 
30: Return H_overlay
```

**Algorithm 4: API Request Processing**

```
Input: HTTP Request with Image File (Request)
Output: JSON Response (Response)
1:  ImageFile ← ExtractFile(Request)
2:  ImageBytes ← ReadBytes(ImageFile)
3:  OriginalImage ← PIL.Image.open(ImageBytes)
4:  
5:  // Preprocessing
6:  I_processed ← PreprocessImage(OriginalImage)  // Algorithm 1
7:  
8:  // Inference
9:  Probabilities ← Model.predict(I_processed)  // Algorithm 2
10: 
11: // Format predictions
12: PredictionsList ← []
13: For each (className, prob) in zip(ClassNames, Probabilities):
14:   Append({"class_name": className, "probability": prob})
15: Sort(PredictionsList, key=probability, reverse=True)
16: 
17: // Generate explanation (if requested)
18: If Request.endpoint == "/predict-with-explanation":
19:   HeatmapOverlay ← GenerateGradCAM(Model, OriginalImage)  // Algorithm 3
20:   HeatmapBase64 ← EncodeToBase64(HeatmapOverlay)
21:   Response ← {
22:     "predictions": PredictionsList,
23:     "explanation": {
24:       "type": "grad-cam",
25:       "image_base64": HeatmapBase64,
26:       "format": "png"
27:     }
28:   }
29: Else:
30:   Response ← {"predictions": PredictionsList}
31: 
32: Return JSON(Response)
```


