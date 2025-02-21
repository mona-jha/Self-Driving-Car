# Self-Driving-Car
# Autonomous Driving Perception System

This project implements a comprehensive perception system for autonomous driving using deep learning models for road lane segmentation, object detection, and steering angle prediction.

## Project Components

### 1. Road Lane Segmentation
- Implementation using YOLOv11 for semantic segmentation
- Detects and segments road lanes in real-time
- File: `yolov11-road-lane-segmentation.ipynb`

### 2. Object Detection & Road Visualization
- YOLOv11-based object detection system
- Detects vehicles, pedestrians, traffic signs, and other road objects
- Includes visualization tools for detected objects
- File: `yolov11-object-detection-road-visualization.ipynb`

### 3. Steering Angle Prediction
- Keras-based deep learning model
- Predicts optimal steering angles based on road conditions
- End-to-end learning approach
- File: `keras-model-steering-angle-prediction.ipynb`

## Setup Requirements
- Python 3.8+
- PyTorch
- TensorFlow/Keras
- OpenCV
- CUDA-capable GPU (recommended)

## Usage

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the notebooks in the following order:
   - Lane segmentation
   - Object detection
   - Steering angle prediction

## Model Architecture
- Lane Segmentation: YOLOv11 backbone with custom segmentation head
- Object Detection: YOLOv11 with pretrained weights
- Steering Angle: CNN-LSTM hybrid network

## Performance Metrics
- Lane Segmentation: mIoU > 0.85
- Object Detection: mAP > 0.75
- Steering Angle: MSE < 0.1

## License
MIT License

## Contributors
- Project Team Members

