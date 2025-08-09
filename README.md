# Deepfake Detection with XAI (Perturbation-based SHAP)

This project applies **Explainable AI (XAI)** techniques to understand how a deepfake detection model makes its predictions.  
We used a **TensorFlow Lite (TFLite)** model combining **ResNet + LSTM** architecture for video-based classification.  
The input shape of the model is `(1, 10, 224, 224, 3)`, representing 10 video frames of size 224x224 pixels with 3 color channels.  

## Key Features:
- **Deepfake Video Classification**: Detects whether a given video is "REAL" or "FAKE".
- **Perturbation-based SHAP Explanation**: Divides each video frame into regions and measures how removing each region affects the model's prediction score.
- **Heatmap Visualization**: Highlights the most influential spatial regions in the video frames.
- **High-Resolution Analysis**: Supports both coarse (4x4) and fine (16x16) grid resolutions for deeper insight.

## Technologies Used:
- **TensorFlow & TFLite** for model loading and inference.
- **NumPy & OpenCV** for frame preprocessing and manipulation.
- **Matplotlib & Seaborn** for result visualization.
- **Python** (Jupyter Notebook) for implementation.
