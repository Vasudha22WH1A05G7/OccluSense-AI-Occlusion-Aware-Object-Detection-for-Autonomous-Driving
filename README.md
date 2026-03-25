# OccluSense AI - Occlusion-Aware Object Detection for Autonomous Driving

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Django-5.0+-green.svg" alt="Django">
  <img src="https://img.shields.io/badge/YOLOv8-ultralytics-orange.svg" alt="YOLO">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
</p>

A Django-based web application for occlusion-aware object detection in autonomous driving scenarios. Features custom NMS (Non-Maximum Suppression) implementations including **Soft-NMS** and **DIoU-NMS** for superior handling of occluded objects.

## ✨ Features

- **Pre-trained YOLOv8 Model** - No training required, uses COCO-pretrained weights
- **Three NMS Methods**:
  - Standard NMS - Fast, traditional hard suppression
  - Soft-NMS - Gaussian decay for better crowd handling
  - DIoU-NMS - Distance-aware suppression for occlusion scenarios
- **Multiple Input Sources**: Image upload, video upload, real-time webcam
- **Side-by-side NMS Comparison** - Compare all methods on the same image
- **Interactive Dashboard** - Chart.js analytics and performance metrics
- **REST API** - Full API for integration with other systems
- **Beautiful UI** - Modern dark theme with glassmorphism effects

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Webcam (optional, for real-time detection)

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd "Occlusion aware Object Detection for SelfDriving Cars"
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Apply database migrations**:
   ```bash
   python manage.py makemigrations detection
   python manage.py migrate
   ```

5. **Create a superuser (optional)**:
   ```bash
   python manage.py createsuperuser
   ```

6. **Run the development server**:
   ```bash
   python manage.py runserver
   ```

7. **Open your browser** and navigate to: http://127.0.0.1:8000/

## 📁 Project Structure

```
Occlusion aware Object Detection for SelfDriving Cars/
├── manage.py
├── requirements.txt
├── README.md
│
├── occlusion_detection/          # Main Django project
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── detection/                    # Core detection app
│   ├── models.py                 # Database models
│   ├── views.py                  # Web views
│   ├── urls.py                   # URL routing
│   ├── admin.py                  # Admin configuration
│   ├── utils/
│   │   ├── nms.py               # Custom NMS implementations
│   │   ├── inference.py         # YOLO detector wrapper
│   │   └── metrics.py           # Performance metrics
│   └── templates/detection/
│       ├── base.html            # Base template
│       ├── home.html            # Landing page
│       ├── upload.html          # File upload
│       ├── results.html         # Detection results
│       ├── dashboard.html       # Analytics dashboard
│       ├── compare.html         # NMS comparison
│       └── webcam.html          # Real-time detection
│
├── api/                          # REST API app
│   ├── views.py                  # API endpoints
│   └── urls.py
│
├── results/                      # Results browsing app
│   ├── views.py
│   └── urls.py
│
├── static/
│   ├── css/style.css            # Custom styling
│   └── js/main.js               # JavaScript utilities
│
└── media/                        # User uploads & results
    ├── uploads/
    └── results/
```

## 🎯 Usage

### Web Interface

1. **Upload Page** (`/upload/`)
   - Drag and drop or click to upload images/videos
   - Configure NMS method, confidence threshold, IoU threshold
   - Select target classes (cars, pedestrians, bikes, etc.)

2. **Results Page** (`/results/<id>/`)
   - View annotated images with bounding boxes
   - See detection table with confidence scores
   - Analyze class distribution with Chart.js

3. **Compare NMS** (`/compare/`)
   - Upload an image and compare all three NMS methods
   - Side-by-side visualization
   - Performance comparison charts

4. **Dashboard** (`/dashboard/`)
   - Performance metrics over time
   - NMS method usage statistics
   - Class distribution analytics

5. **Webcam** (`/webcam/`)
   - Real-time detection from webcam
   - Adjustable settings during runtime
   - Live FPS and detection count

### REST API

```bash
# Upload image for detection
curl -X POST -F "file=@image.jpg" -F "nms_method=soft" \
     http://127.0.0.1:8000/api/detect/

# Get detection results
curl http://127.0.0.1:8000/api/results/1/

# Get performance metrics
curl http://127.0.0.1:8000/api/metrics/

# Compare NMS methods
curl -X POST -F "file=@image.jpg" \
     http://127.0.0.1:8000/api/compare-nms/
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detect/` | POST | Upload image/video for detection |
| `/api/detect/realtime/` | POST | Process base64 frame (webcam) |
| `/api/results/<id>/` | GET | Get detection results by session ID |
| `/api/metrics/` | GET | Get aggregated performance metrics |
| `/api/compare-nms/` | POST | Compare all NMS methods on one image |

## ⚙️ Configuration

### Detection Settings (in `settings.py`)

```python
DETECTION_CONFIG = {
    'MODEL_PATH': 'yolov8n.pt',           # Auto-downloads
    'DEFAULT_CONFIDENCE_THRESHOLD': 0.25,
    'DEFAULT_IOU_THRESHOLD': 0.45,
    'DEFAULT_NMS_METHOD': 'standard',
    'SOFT_NMS_SIGMA': 0.5,
    'TARGET_CLASSES': [0, 1, 2, 3, 5, 7, 9],  # COCO class IDs
}
```

### Target Classes (COCO)

| Class ID | Name |
|----------|------|
| 0 | person |
| 1 | bicycle |
| 2 | car |
| 3 | motorcycle |
| 5 | bus |
| 7 | truck |
| 9 | traffic light |

## 🔬 NMS Methods Explained

### Standard NMS
Traditional Non-Maximum Suppression with hard thresholding. Boxes with IoU > threshold are removed.

```
if IoU(box_i, box_j) > threshold:
    remove box_j
```

### Soft-NMS
Reduces confidence scores using Gaussian decay instead of hard removal. Better for crowded scenes.

```
score_j = score_j * exp(-(IoU^2) / sigma)
```

### DIoU-NMS
Considers center point distance in addition to IoU. Better for occluded objects with distant centers.

```
DIoU = IoU - (center_distance^2 / diagonal^2)
```

## 📊 Output Format

Detection results are returned as JSON:

```json
{
    "detections": [
        {
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.89,
            "bbox": [100, 200, 300, 400],
            "is_occluded": false,
            "occlusion_score": 0.12
        }
    ],
    "num_detections": 5,
    "nms_method": "soft",
    "processing_time_ms": 45.2,
    "fps": 22.1,
    "annotated_image_url": "/media/results/result_1.jpg"
}
```

## 🛠️ Development

### Running Tests
```bash
python manage.py test detection
```

### Creating Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### Admin Panel
Access at http://127.0.0.1:8000/admin/ to manage:
- Uploaded media
- Detection sessions
- Detection results
- Performance metrics

## 📝 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Pre-trained object detection models
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Django](https://www.djangoproject.com/) - Web framework
- [Chart.js](https://www.chartjs.org/) - JavaScript charting library
- [Bootstrap 5](https://getbootstrap.com/) - CSS framework
