
# Deepfake Detection + Heatmap Analysis

## 📌 Overview
This project detects deepfake videos using a pre-trained TensorFlow Lite (TFLite) model and applies explainable AI techniques to understand which frames and image regions influence the prediction.

---

## 1️⃣ Loading the Project and Libraries
- **Clone** the repository:
```bash
!git clone https://github.com/RishiPratap/Deep-fake-Detection-app.git
%cd Deep-fake-Detection-app
```
- Install required libraries (`numpy`, `opencv`, `pillow`, `tensorflow`).
- Download the TFLite model.

---

## 2️⃣ Load and Prepare the Model
```python
from tensorflow.lite.python.interpreter import Interpreter

interpreter = Interpreter(model_path="deepfake_detection_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```

---

## 3️⃣ Video Upload and Frame Extraction
- Upload a video.
- Extract 10 evenly spaced frames (`extract_frames` function).
- **Preprocessing**:
  - Resize to `(224,224)`.
  - Convert BGR → RGB.
  - Normalize to [0,1].
  - Add batch dimension.

---

## 4️⃣ Basic Prediction
- Pass frames to the model.
- Output:
  - >0.5 → **FAKE**
  - ≤0.5 → **REAL**
- Display frames with predictions.

---

## 5️⃣ Frame Importance Analysis
- Loop over each frame.
- Replace with zeros.
- Measure score difference (`diff`).
- Output: List of importance per frame.

---

## 6️⃣ Spatial Importance (Grid-based Heatmap)
- Choose a frame (first or most important).
- Split into grid (`grid_size x grid_size`):
  - Started with **4x4**.
  - Increased to **16x16** for higher resolution.
- For each grid cell:
  - Zero out region.
  - Measure change in output.
  - Store in `importance_map`.
- Plot with `seaborn.heatmap`.
- Overlay on the original frame.

---

## 7️⃣ Observations & Improvements
- Model sometimes focuses on non-face areas (clothing, background).
- Improvements:
  - Apply **face cropping** before analysis.
  - Use face-focused deepfake detection models.
  - Train with better datasets to reduce bias.

---

## 📂 Output Files
- Heatmaps for each grid size.
- Frame prediction images.
- Frame importance list.

---

## 🛠 Requirements
- Python 3.x
- TensorFlow Lite
- OpenCV
- NumPy
- Matplotlib
- Seaborn
