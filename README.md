## Blood Cancer Detection Backend (Hybrid CNN + Explainable AI)

This project is a **backend-only** implementation for blood cancer detection from microscopic blood smear images using:

- **Hybrid Convolutional Neural Network (CNN)** for image classification
- **Explainable AI (XAI)** using **Grad-CAM** heatmaps
- **FastAPI** backend for model inference and explanation APIs

### 1. Project Structure

- **`requirements.txt`**: Python dependencies
- **`model.py`**: Hybrid CNN model definition
- **`xai.py`**: Grad-CAM based explainability utilities
- **`train.py`**: Script to train and save the model
- **`api.py`**: FastAPI app exposing prediction and explanation endpoints
- **`main.py`**: Entry point to run the backend server

You can extend or modify this structure as needed for your minor project report.

### 2. Dataset Format (Blood Cancer Images)

Prepare your dataset as image folders, for example:

```text
data/
  train/
    normal/
      img1.png
      img2.png
      ...
    cancer/
      img3.png
      ...
  val/
    normal/
    cancer/
```

- Each subfolder inside `train` / `val` represents one class (e.g. `normal`, `cancer`, or more detailed leukemia subtypes).
- You can adapt folder names to your own labels; they will be discovered automatically.

### 3. Training the Hybrid CNN

1. Install dependencies (ideally in a virtual environment):

```bash
pip install -r requirements.txt
```

2. Place your dataset under the `data/` directory as described above.

3. Run training:

```bash
python train.py
```

This will:

- Build and train the **hybrid CNN** model
- Save the best model as `hybrid_cnn_blood_cancer.h5`
- Save the class names in `class_names.txt`

### 4. Running the Backend Server

After training (or after placing a pre-trained model in the project folder):

```bash
python main.py
```

The FastAPI server will start (by default on `http://127.0.0.1:8000`).

Key endpoints:

- `POST /predict`  
  - Input: image file (form-data)  
  - Output: predicted class probabilities  
- `POST /predict-with-explanation`  
  - Input: image file (form-data)  
  - Output: predicted class probabilities **and** Grad-CAM heatmap (as base64 PNG)

You can also open the interactive API docs:

- Swagger UI: `http://127.0.0.1:8000/docs`

### 5. Hybrid CNN and Explainable AI (Concept Summary)

- **Hybrid CNN** in this project combines:
  - Standard convolution + max pooling block
  - A multi-branch convolution block with multiple kernel sizes (Inception-style)
  - Depthwise separable convolution block (more efficient and expressive)
- **Explainable AI (Grad-CAM)**:
  - Computes gradients of the predicted class score with respect to the last convolutional feature maps
  - Produces a heatmap showing which image regions contributed most to the decision
  - Overlays the heatmap on the original image to visually explain the prediction

You can describe and illustrate these components in your project report and presentations.


