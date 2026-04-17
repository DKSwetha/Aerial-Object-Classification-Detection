# Aerial Object Detection & Classification (Bird vs Drone)

## Overview

This project focuses on detecting and classifying aerial objects as **Birds or Drones ** using deep learning techniques.

It combines:

* Image Classification (CNN + Transfer Learning)
* Object Detection using YOLOv8
* Model comparison and evaluation
* Streamlit-based deployment

---

## Technologies Used

* Python
* TensorFlow / Keras
* YOLOv8 (Ultralytics)
* OpenCV
* NumPy, Matplotlib
* Streamlit

---

## Project Structure

```
├── custom_cnn.py
├── transfer_learning.py
├── yolov8_pipeline.py
├── model_comparison.py
├── app.py (Streamlit deployment)
├── data_preprocessing.py
├── README.md
```

---

## 🧠 Models Used

### 1. Custom CNN

* Built from scratch
* Uses Conv2D, MaxPooling, Dropout

### 2. Transfer Learning Models

* MobileNetV2 (Best performer)
* ResNet50
* EfficientNetB0

### 3. YOLOv8

* Used for real-time object detection
* Detects bounding boxes + class labels

---

## Results Summary

| Model          | Accuracy | F1 Score         |
| -------------- | -------- | ---------------- |
| Custom CNN     | ~82%     | ~0.76            |
| ResNet50       | ~86%     | ~0.83            |
| MobileNetV2    | **~98%** | **~0.98 (Best)** |
| EfficientNetB0 | Poor     | ~0.16            |

### YOLO Results:

* mAP@50 ≈ **0.82**
* Strong drone detection
* Moderate bird detection

---

## Key Insights

* Transfer learning significantly outperformed custom CNN
* MobileNetV2 achieved best balance of precision & recall
* YOLO showed strong detection performance but struggled with small objects and background confusion

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run training

```bash
python cnn_model.py
python transfer_learning.py
python yolo_training.py
```

### 3. Run Streamlit app

```bash
streamlit run app.py
```

## Results

### Model Performance
- Confusion Matrix (Best Model)
  
<img width="900" height="750" alt="MobileNetV2_confusion_matrix" src="https://github.com/user-attachments/assets/7a851a1d-80a7-495d-be10-df22de5d92cf" />

  


- Confusion Matrix (YOLO)
  
 <img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/058ca553-98ce-4bae-8eb0-4db42cc8ede5" />



### Training Behavior
- Accuracy & Loss Graph
<img width="2100" height="750" alt="MobileNetV2_finetuned_curves" src="https://github.com/user-attachments/assets/409255ad-8bd0-4330-ad3e-0bc1ecff8124" />


### Model Comparison
- Bar Chart (All Metrics)
<img width="2100" height="900" alt="model_comparison_bar" src="https://github.com/user-attachments/assets/a360fc27-502f-4931-ae5b-916cb751f424" />


### YOLO Detection Results
- Sample bounding box outputs
<img width="2250" height="1500" alt="yolo_sample_predictions" src="https://github.com/user-attachments/assets/4b1fb800-ed77-40a6-b5c6-aa2175711d09" />


## Future Improvements

* Improve bird detection recall
* Handle class imbalance
* Optimize YOLO for small object detection
* Deploy on edge devices (Jetson Nano, etc.)

---

## Conclusion

This project demonstrates an end-to-end deep learning pipeline combining classification, detection, evaluation, and deployment for real-world aerial object analysis.
