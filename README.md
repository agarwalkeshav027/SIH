# WildCare — Intelligent Coexistence AI

### AI-Powered Human–Wildlife Conflict Prevention and Early-Warning System

WildCare is an AI-driven wildlife monitoring and early-warning platform designed to detect wildlife near human-populated or high-risk areas, generate alerts, analyze environmental conditions, and provide movement-risk information through a centralized command dashboard.

The system combines **edge AI computer vision, environmental data, movement analysis, cloud APIs, and an interactive monitoring dashboard** to support faster responses to potential human–wildlife conflict situations.

---

## Problem

Human–wildlife encounters can become dangerous when animals move through areas shared with people, roads, settlements, or other infrastructure.

Traditional monitoring approaches can be:

- Dependent on manual observation.
- Slow to react to animal movement.
- Difficult to scale across multiple monitoring locations.
- Limited in their ability to combine visual detection with environmental context.
- Unable to provide a unified operational view of detected incidents.

WildCare addresses this by combining real-time visual detection with alert management, geographic information, environmental features, and movement analysis.

---

## Solution

WildCare creates a pipeline from **wildlife detection → verification → alert generation → environmental analysis → movement prediction → centralized monitoring**.

The system is designed around the following major components:

```text
                    Camera / Sensor Node
                           |
                           v
                 +--------------------+
                 |     Edge AI        |
                 |   YOLO Detection   |
                 +--------------------+
                           |
                 Wildlife Detection
                           |
                           v
                 Human Detection Filter
                           |
                           v
                  Human/Wildlife Check
                           |
                           v
                  +----------------+
                  | Alert Dispatch |
                  +----------------+
                           |
                           v
                 +--------------------+
                 |  Cloud Backend    |
                 | FastAPI + SQLite  |
                 +--------------------+
                           |
                           v
              +-------------------------+
              | WildCare Command Centre |
              |      Streamlit UI       |
              +-------------------------+
                    /     |      \
                   /      |       \
                  v       v        v
              Alerts    Map     Analytics
                           |
                           v
                 Movement Prediction
                           |
                           v
                Environmental Context
                           |
                           v
                 Risk-Aware Monitoring
```

---

# Key Features

## Edge AI Wildlife Detection

The edge inference pipeline uses a fine-tuned **YOLOv8** model to identify wildlife from camera frames.

The current cleaned dataset contains 16 wildlife classes:

- Zebra
- Lion
- Leopard
- Cheetah
- Tiger
- Bear
- Bull
- Elephant
- Horse
- Fox
- Kangaroo
- Deer
- Hippopotamus
- BrownBear
- Rhinoceros
- Jaguar

The project also uses a YOLOv8 model trained on the COCO dataset to identify humans. citeturn4view6turn4view5turn6view0

---

## Human Detection and False-Positive Suppression

The edge inference pipeline performs human detection separately from wildlife detection.

When a wildlife bounding box overlaps significantly with a detected human bounding box, the wildlife detection can be suppressed.

The implementation uses **Intersection over Union (IoU)** for this comparison.

This is useful for reducing false alerts in situations where:

- A ranger is present near an animal.
- A person appears inside a wildlife detection region.
- The camera scene contains both humans and animals.

The current implementation uses an IoU threshold of `0.35` for suppression. citeturn6view0

---

## Real-Time Camera Inference

The edge system can process frames from a webcam using OpenCV.

The inference pipeline:

1. Captures a frame.
2. Runs human detection.
3. Runs wildlife detection.
4. Compares wildlife detections with human detections.
5. Identifies genuine wildlife detections.
6. Draws detection bounding boxes.
7. Generates an alert when appropriate.
8. Sends the alert to the cloud backend.

The wildlife confidence threshold is currently `0.55`, while the human detection threshold is `0.50`. citeturn6view0

---

## Alert Cooldown

To prevent continuous repeated alerts from the same detection event, the edge inference system uses a cooldown period.

The current cooldown is:

```text
5 seconds
```

Only after the cooldown has elapsed can another detection trigger an alert. citeturn5view0

---

# Cloud Backend

The cloud backend is implemented using **FastAPI** and **SQLAlchemy**.

It stores wildlife detection events in a SQLite database.

Each detection record contains:

```text
id
node_id
species
confidence
latitude
longitude
timestamp
status
```

The backend exposes REST API endpoints for receiving detections, retrieving alerts, and updating incident status. citeturn3view0

---

## API Endpoints

### Create Alert

```http
POST /api/alerts
```

Receives an alert generated by an edge node.

Example payload:

```json
{
  "node_id": "Highway-Node-001",
  "species": "Tiger",
  "confidence": 0.91,
  "latitude": 19.231,
  "longitude": 72.825,
  "timestamp": "2026-01-01T12:00:00Z"
}
```

The backend stores the detection with an initial status of:

```text
ACTIVE
```

---

### Get Alerts

```http
GET /api/alerts
```

Returns the latest 50 detection records for the monitoring dashboard. citeturn4view4

---

### Update Alert Status

```http
PATCH /api/alerts/{alert_id}/status
```

Supported statuses are:

```text
ACTIVE
ACKNOWLEDGED
RESOLVED
```

This allows incidents to be managed from the dashboard. citeturn4view4

---

# Environmental Intelligence

The project includes a habitat service that attempts to obtain environmental information for a given geographic coordinate.

The environmental features include:

- NDVI
- Terrain slope
- Distance to water

When Google Earth Engine is available, the service uses:

### Sentinel-2

For NDVI calculation.

### SRTM

For terrain slope information.

If Earth Engine is unavailable, the implementation falls back to deterministic spatial calculations so that the system can continue providing values during local/offline operation. citeturn7view0

---

# Wildlife Movement Prediction

The project contains a predictive engine for interpreting potential wildlife movement.

The `WildlifePredictiveEngine` uses a Random Forest classifier with environmental and temporal features:

```text
Hour
NDVI
Distance to Water
Slope
Is Carnivore
```

The model predicts a movement reason from categories such as:

```text
Water Source Seeking
Canopy & Forage Migration
Human Avoidance Movement
Territorial Patrol
```

The current implementation initializes the Random Forest using a small baseline dataset representing these behavioral patterns. citeturn5view1turn6view2

---

## Movement Trajectory and Corridors

The predictive engine also provides:

- Current location
- Predicted next location
- Potential movement corridors

The current trajectory implementation projects a future coordinate using a spatial calculation.

When sufficient historical coordinates are available, **DBSCAN clustering** is used to identify groups of historical movement coordinates and calculate corridor centroids. citeturn6view2

---

# WildCare Command Centre

The project includes an interactive Streamlit dashboard branded as:

**WildCare — Intelligent Coexistence AI**

The interface provides several operational sections:

- Command Centre
- Detection Console
- Active Alerts
- Movement Map
- Analytics & Logs

The dashboard is designed as a centralized monitoring interface for wildlife detections and response information. citeturn7view2

---

# System Architecture

```text
                    ┌─────────────────────┐
                    │    Camera / Node    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Edge AI Model    │
                    │      YOLOv8         │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Human Detection +   │
                    │ IoU Verification    │
                    └──────────┬──────────┘
                               │
                         Wildlife Found
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Alert Generator   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   FastAPI Backend   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     SQLite DB       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Dashboard │
                    └──────┬───────┬──────┘
                           │       │
              ┌────────────┘       └────────────┐
              ▼                                 ▼
     ┌──────────────────┐             ┌──────────────────┐
     │ Movement Engine  │             │ Habitat Service  │
     │ Random Forest +  │             │ NDVI + Slope +   │
     │ DBSCAN            │             │ Water Proximity  │
     └──────────────────┘             └──────────────────┘
```

---

# Machine Learning Pipeline

## Dataset Preparation

The repository contains an original wildlife dataset configuration with 54 classes.

A dataset-cleaning pipeline filters this dataset down to 16 target wildlife classes and remaps the original class IDs to the new class IDs.

The mapping includes species such as:

```text
Zebra
Lion
Leopard
Cheetah
Tiger
Bear
Bull
Elephant
Horse
Fox
Kangaroo
Deer
Hippopotamus
BrownBear
Rhinoceros
Jaguar
```

The cleaning process operates across:

```text
train
valid
test
```

and creates a cleaned dataset suitable for YOLO training. citeturn2view6turn4view6

---

## YOLO Training

The repository contains a training script based on the Ultralytics YOLO framework.

The cleaned model is trained using:

```text
Model: YOLOv8 Nano
Epochs: 30
Image Size: 640
Batch Size: 16
Device: GPU
Workers: 2
```

The training configuration uses the cleaned 16-class dataset. citeturn7view7

---

# Project Structure

```text
SIH/
│
├── SIH_Wildlife/
│   └── edge_prototype/
│       ├── weights/
│       ├── BoxF1_curve.png
│       ├── BoxPR_curve.png
│       ├── BoxP_curve.png
│       ├── BoxR_curve.png
│       ├── confusion_matrix.png
│       ├── confusion_matrix_normalized.png
│       ├── labels.jpg
│       ├── results.csv
│       └── results.png
│
├── cloud_backend/
│   ├── api.py
│   └── alerts.db
│
├── dataset/
│   ├── data.yaml
│   └── filtered_data.yaml
│
├── edge_ai/
│   ├── fix_dataset.py
│   ├── repair_checkpoint.py
│   ├── run_inference.py
│   ├── train_edge.py
│   └── yolov8n.pt
│
├── frontend/
│   ├── assets/
│   ├── satellite_engine/
│   │   └── habitat_service.py
│   ├── app.py
│   └── app_backup.py
│
├── ml_engine/
│   └── movement_predictor.py
│
├── runs/
│   └── detect/
│       └── SIH_Wildlife/
│
├── fix_dataset.py
├── requirements.txt
├── test_model.py
├── train_clean.py
├── yolo26n.pt
└── yolov8n.pt
```

---

# Components

## `edge_ai/`

Responsible for edge-side computer vision.

### `run_inference.py`

Runs real-time camera inference using:

- OpenCV
- Custom YOLO wildlife model
- YOLOv8n human detector
- IoU-based human/wildlife suppression
- Alert dispatch through HTTP

### `train_edge.py`

Handles GPU-based continuation/fine-tuning of the edge model.

### `repair_checkpoint.py`

Repairs a stored training checkpoint by replacing the scaler state with a valid GradScaler state before saving the repaired checkpoint.

### `fix_dataset.py`

Performs wildlife dataset class filtering and label remapping.

---

## `cloud_backend/`

Contains the FastAPI-based alert backend.

### `api.py`

Responsible for:

- Database initialization
- Alert ingestion
- Alert retrieval
- Incident status updates

The database is currently SQLite-based.

---

## `ml_engine/`

Contains the wildlife movement intelligence layer.

### `movement_predictor.py`

Provides:

- Behavioral reason prediction
- Random Forest classification
- Carnivore identification
- Future coordinate projection
- Historical movement clustering using DBSCAN

---

## `frontend/`

Contains the Streamlit monitoring application.

### `app.py`

Provides the WildCare command centre and integrates the monitoring interface with the backend and prediction components.

### `satellite_engine/habitat_service.py`

Provides habitat-related environmental features such as NDVI, slope, and water proximity.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/agarwalkeshav027/SIH.git
cd SIH
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Backend

Start the FastAPI backend:

```bash
uvicorn cloud_backend.api:app --reload
```

The API will then be available locally through the configured FastAPI server.

FastAPI also provides its interactive API documentation through its standard documentation endpoints.

---

# Running the Frontend

Start the Streamlit dashboard:

```bash
streamlit run frontend/app.py
```

The WildCare Command Centre will open in your browser.

---

# Running Edge AI

The edge inference system requires:

- A working camera/webcam.
- Python environment with the required computer-vision dependencies.
- Ultralytics YOLO.
- The trained wildlife checkpoint.
- YOLOv8n weights for human detection.

Run:

```bash
python edge_ai/run_inference.py
```

The inference script captures frames from the default camera and sends generated alerts to:

```text
http://localhost:8000/api/alerts
```

The current implementation also writes the latest detection frame into the frontend assets directory. citeturn5view0turn6view1

---

# Training the Wildlife Model

The repository includes scripts for preparing and training the wildlife detector.

First prepare the cleaned dataset:

```bash
python fix_dataset.py
```

Then train:

```bash
python train_clean.py
```

The training script uses the cleaned dataset configuration:

```text
./dataset_cleaned/wildlife_16.yaml
```

and fine-tunes a YOLOv8 Nano model. citeturn2view6turn7view7

---

# Data Flow

A typical detection event follows this sequence:

```text
Camera Frame
     |
     v
Human Detection
     |
     v
Wildlife Detection
     |
     v
IoU-Based Verification
     |
     v
Species + Confidence
     |
     v
Location + Timestamp
     |
     v
POST /api/alerts
     |
     v
SQLite Database
     |
     v
Dashboard
     |
     +----> Alert Management
     |
     +----> Movement Analysis
     |
     +----> Habitat Information
     |
     +----> Analytics
```

---

# Technology Stack

## Artificial Intelligence

- YOLOv8
- Ultralytics
- Random Forest
- DBSCAN
- OpenCV

## Backend

- Python
- FastAPI
- Pydantic
- SQLAlchemy
- SQLite

## Frontend

- Streamlit
- HTML/CSS styling
- Interactive monitoring components

## Geospatial / Environmental Data

- Google Earth Engine
- Sentinel-2
- SRTM

## Development

- Python
- Git
- GitHub

---

# Current Wildlife Classes

The cleaned training configuration contains 16 classes:

| ID | Species |
|---:|---|
| 0 | Zebra |
| 1 | Lion |
| 2 | Leopard |
| 3 | Cheetah |
| 4 | Tiger |
| 5 | Bear |
| 6 | Bull |
| 7 | Elephant |
| 8 | Horse |
| 9 | Fox |
| 10 | Kangaroo |
| 11 | Deer |
| 12 | Hippopotamus |
| 13 | BrownBear |
| 14 | Rhinoceros |
| 15 | Jaguar |

---

# Example Alert

A detection generated by an edge node can be represented as:

```json
{
  "node_id": "Highway-Node-001",
  "species": "Tiger",
  "confidence": 0.91,
  "latitude": 19.231,
  "longitude": 72.825,
  "timestamp": "2026-01-01T12:00:00Z"
}
```

After being received by the backend, the alert is stored and initially marked as:

```text
ACTIVE
```

The dashboard can subsequently change the incident state to:

```text
ACKNOWLEDGED
```

or:

```text
RESOLVED
```

---

# Design Goals

WildCare is built around several principles:

### Early Detection

Identify wildlife presence as close to the edge as possible.

### Rapid Alerting

Send detections to the backend immediately after a qualifying detection.

### Context-Aware Monitoring

Combine detection information with geographic and environmental features.

### Centralized Operations

Provide a single dashboard for alerts, detections, movement information, and analytics.

### Reduced False Alerts

Use human detection and IoU-based suppression to distinguish wildlife detections from overlapping human/ranger detections.

### Scalable Architecture

Separate the edge AI, backend, frontend, and predictive components so that they can evolve independently.

---

# Future Improvements

Potential improvements to the current prototype include:

- Deployment on dedicated edge hardware such as NVIDIA Jetson devices.
- Real GPS integration for field sensor nodes.
- Multi-camera sensor networks.
- Real-time notification services such as SMS, email, or mobile push notifications.
- Historical movement datasets for training the movement model.
- More robust trajectory forecasting using time-series or sequence models.
- Improved geospatial corridor analysis.
- Automated risk scoring based on population density, roads, settlements, and historical encounters.
- Cloud-hosted database infrastructure for multi-node deployments.
- Authentication and role-based access for forest officials.
- Model quantization and optimization for low-power edge devices.
- Automated model evaluation and retraining pipelines.

---

# Project Status

This repository represents a working prototype combining the major components required for an AI-assisted human–wildlife conflict monitoring platform.

The current implementation includes functional components for:

- Wildlife detection
- Human detection
- Edge inference
- Alert generation
- REST-based alert ingestion
- SQLite alert storage
- Dashboard visualization
- Habitat feature extraction
- Wildlife movement reasoning
- Movement corridor clustering
- YOLO dataset preparation and training

---
