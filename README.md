# Object-Detection-Using-YOLOv5


## Introduction

This project leverages the power of the YOLOv5 model for object detection in images. The goal is to create an API service that allows users to upload images and get back predictions, i.e., the detected objects within those images. The project is implemented using **FastAPI** for the API, and the **YOLOv5** pre-trained model is used to detect objects in the uploaded images.

---

## Model Details

### YOLOv5 (You Only Look Once Version 5)

- **Model Type**: Object Detection
- **Framework**: PyTorch
- **Pre-trained Model**: YOLOv5 (YOLOv5s.pt)
- **Purpose**: Detect objects in images with high accuracy and speed.
- **Model Features**:
  - **Single Shot Detection**: YOLOv5 is an advanced single-shot detector that can predict multiple objects in an image in real-time.
  - **Pretrained Weights**: This model is trained on the **COCO dataset**, capable of detecting 80 object classes.
  - **Speed and Accuracy**: YOLOv5 is known for its balance between speed and accuracy, which makes it suitable for various real-time applications.

For more details about YOLOv5, check out the official [YOLOv5 repository](https://github.com/ultralytics/yolov5).

---

## App Architecture

The application is a **FastAPI** implementation that uses the YOLOv5 model to provide object detection capabilities through an HTTP API. 

### Architecture Components:
1. **FastAPI**:
   - The core of the application, providing the endpoints to interact with the YOLOv5 model.
   - It handles image uploads, invokes the model for predictions, and returns the results in a user-friendly format.
  
2. **YOLOv5 Model**:
   - The pre-trained YOLOv5 model (`model.pt`) is loaded into memory for making predictions.
   - The model is used to detect objects within the uploaded images and returns a list of predicted classes, bounding boxes, and confidence scores.
  
3. **API Endpoints**:
   - **POST /predict**: Accepts an image file and returns the predicted objects with their bounding boxes and labels.
   - **GET /**: A simple health check endpoint to verify that the API is running.

4. **Features**:
   - **Image Upload**: Users can upload an image via the API for analysis.
   - **Prediction Results**: The model returns a JSON object containing detected objects' classes, confidence scores, and bounding box coordinates.
   - **Multiple Objects**: The model can detect multiple objects within a single image and label them accordingly.

---

## Installation Instructions

### Prerequisites:
- Python 3.7 or higher
- Pip (Package Installer for Python)
- Git (for cloning the repository)

### Steps to Setup the Project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/7Amulya/Object-Detection-Using-YOLOv5.git
