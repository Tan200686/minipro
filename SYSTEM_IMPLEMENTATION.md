# SYSTEM IMPLEMENTATION

## Model Architecture Design

The application leverages a **Hybrid Convolutional Neural Network (CNN)** architecture specifically designed to capture multi-scale features from blood smear images. The model is organized into three primary blocks to streamline feature extraction and classification. The **Standard Convolution Block** performs initial feature extraction using traditional 3×3 convolutions with batch normalization and max pooling. The **Multi-Branch Convolution Block** utilizes an Inception-style architecture with parallel branches of 1×1, 3×3, and 5×5 kernels; this hybrid approach enables the model to capture both fine-grained cellular details and broader morphological patterns simultaneously. Finally, the **Depthwise Separable Convolution Block** provides efficient, deeper feature representation while reducing computational overhead. The classification head utilizes global average pooling followed by dropout regularization and a softmax output layer to produce probability distributions over cancer classes (e.g., Normal, Cancer, or specific leukemia subtypes).

The **Explainable AI (XAI)** component employs **Gradient-weighted Class Activation Mapping (Grad-CAM)** to generate interpretable visualizations. The Grad-CAM algorithm computes gradients of the top predicted class score with respect to the activations of the last convolutional layer (`last_conv_block`). These gradients are then pooled and weighted to produce a heatmap that highlights the image regions most influential in the model's decision. The heatmap is overlaid on the original blood smear image using a jet colormap, providing clinicians with visual evidence of which cellular structures or morphological features the model identified as indicative of cancer.

## System Architecture

The system follows a **backend-only, stateless API architecture** optimized for medical image inference and explanation generation.

**Client Layer**: External applications (web frontends, mobile apps, or medical imaging systems) communicate with the backend via RESTful HTTP endpoints. Image uploads are handled as multipart form-data, ensuring compatibility with standard HTTP clients.

**API Layer (FastAPI)**: The FastAPI application serves as the primary interface, exposing two core endpoints:
- `/predict`: Performs inference using the hybrid CNN model and returns class probabilities
- `/predict-with-explanation`: Extends prediction with Grad-CAM heatmap generation, returning both probabilities and a base64-encoded PNG visualization

**Model Inference Layer**: The trained hybrid CNN model (`hybrid_cnn_blood_cancer.h5`) is loaded into memory at application startup. Each inference request preprocesses the input image (resize to 224×224, normalize pixel values to [0,1]), passes it through the model, and post-processes the output probabilities.

**Explainability Layer**: The XAI module (`xai.py`) implements Grad-CAM computation in real-time. For each explanation request, it constructs a gradient computation graph, extracts activations from the last convolutional layer, computes weighted gradients, and generates a heatmap overlay. The overlay is encoded as a base64 PNG string for efficient transmission over HTTP.

**Data Persistence**: The system maintains minimal on-disk artifacts:
- `hybrid_cnn_blood_cancer.h5`: Serialized Keras model weights and architecture
- `class_names.txt`: Text file listing class labels in training order
- `training_history.json`: Optional training metrics for model evaluation

**Fig. 2: System Architecture Diagram**

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                        │
│  (Web Frontend / Mobile App / Medical Imaging System)        │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST (TLS 1.3)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend Server                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  API Endpoints:                                        │   │
│  │  • POST /predict                                       │   │
│  │  • POST /predict-with-explanation                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                         │                                     │
│         ┌───────────────┴───────────────┐                    │
│         ▼                               ▼                    │
│  ┌──────────────────┐          ┌──────────────────┐          │
│  │  Model Inference │          │  XAI Module       │          │
│  │  (Hybrid CNN)    │          │  (Grad-CAM)      │          │
│  └──────────────────┘          └──────────────────┘          │
│         │                               │                     │
│         └───────────────┬───────────────┘                    │
│                         ▼                                     │
│              ┌──────────────────────┐                        │
│              │  Preprocessing       │                        │
│              │  (Resize, Normalize)  │                        │
│              └──────────────────────┘                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Artifacts (On-Disk)                       │
│  • hybrid_cnn_blood_cancer.h5                                │
│  • class_names.txt                                           │
│  • training_history.json                                     │
└─────────────────────────────────────────────────────────────┘
```

## Security & Privacy Framework

Given the sensitive nature of medical imaging data, a **"Privacy-by-Design"** approach was adopted throughout the system architecture.

**Local Processing & Data Minimization**: The backend processes images entirely in-memory during inference. Raw image data is never persisted to disk after processing; only the model predictions and explanation visualizations are returned to the client. This ensures that patient blood smear images do not accumulate in server logs or temporary storage.

**Encryption in Transit**: All API communications are secured using **TLS 1.3** encryption. The FastAPI application can be deployed behind a reverse proxy (e.g., Nginx) with SSL/TLS certificates to ensure end-to-end encryption between clients and the server.

**Model Security**: The trained model file (`hybrid_cnn_blood_cancer.h5`) is stored with restricted file permissions. In production deployments, model artifacts should be stored in encrypted volumes or secure cloud storage (e.g., AWS S3 with server-side encryption) and loaded only by authorized processes.

**HIPAA Compliance Strategy**: The system explicitly avoids storing **Protected Health Information (PHI)** such as patient names, medical record numbers, or timestamps. The API accepts only image files without metadata, and responses contain only classification probabilities and visualization overlays. By design, the backend operates as a stateless service that does not maintain patient records or audit logs containing PHI, thereby reducing regulatory compliance burden while maintaining diagnostic utility.

**Input Validation & Error Handling**: The API implements strict input validation to reject non-image files and malformed requests. Image preprocessing includes error handling to gracefully manage corrupted or unsupported image formats, preventing potential security vulnerabilities from malformed inputs.

**CORS Configuration**: Cross-Origin Resource Sharing (CORS) is configured to allow controlled access from authorized frontend domains. In production, the `allow_origins` parameter should be restricted to specific trusted domains rather than using wildcard (`*`) to prevent unauthorized cross-origin requests.

**Audit Trail (Optional)**: For compliance purposes, the system can be extended to log anonymized inference requests (e.g., hash of image, timestamp, predicted class) without storing the actual image data. This enables performance monitoring and model validation while maintaining patient privacy.

