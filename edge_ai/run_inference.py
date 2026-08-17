import os
import cv2
import time
import json
import requests
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
FRONTEND_ASSETS = os.path.join(
    PROJECT_ROOT,
    "frontend",
    "assets"
)

os.makedirs(FRONTEND_ASSETS, exist_ok=True)

LATEST_FRAME_PATH = os.path.join(
    FRONTEND_ASSETS,
    "latest_detection.jpg"
)

CUSTOM_MODEL_PATH = os.path.join(
    PROJECT_ROOT, 
    'runs', 
    'detect', 
    'SIH_Wildlife', 
    'clean_wildlife_model-5', 
    'weights', 
    'best.pt'
)

WILDLIFE_CONF = 0.55   # Wildlife confidence threshold
PERSON_CONF = 0.50     # Human suppression threshold
COOLDOWN_TIME = 5.0

# ==========================================
# 2. INITIALIZE DUAL MODELS
# ==========================================
print("[INFO] Loading Wildlife Custom Model...")
wildlife_model = YOLO(CUSTOM_MODEL_PATH)

print("[INFO] Loading Base Human Filter Model...")
# Base COCO model knows 'person' out-of-the-box
human_filter_model = YOLO('yolov8n.pt') 

print("[INFO] Starting Webcam Feed. Press 'q' to quit.")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

last_alert_time = 0

# Helper function to compute Intersection over Union (IoU)
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# ==========================================
# 3. REAL-TIME INFERENCE LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Check for Humans using Base COCO Model
    human_results = human_filter_model(frame, conf=PERSON_CONF, verbose=False)
    human_boxes = []
    for r in human_results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # Class 0 is 'person' in standard COCO
                hb = list(map(int, box.xyxy[0]))
                human_boxes.append(hb)
                # Draw Blue Box for Human / Ranger
                cv2.rectangle(frame, (hb[0], hb[1]), (hb[2], hb[3]), (255, 165, 0), 2)
                cv2.putText(frame, f"Human/Ranger {float(box.conf[0]):.2f}", (hb[0], hb[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    # 2. Check for Wildlife using Fine-tuned Model
    wildlife_results = wildlife_model(frame, conf=WILDLIFE_CONF, verbose=False)

    animal_detected = False
    detected_species = "Unknown"
    animal_conf = 0.0

    for r in wildlife_results:
        for box in r.boxes:
            wb = list(map(int, box.xyxy[0]))
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            species_name = wildlife_model.names[cls_id]

            # Suppress if this detection overlaps significantly with an identified human
            is_human_overlap = any(compute_iou(wb, hb) > 0.35 for hb in human_boxes)

            if not is_human_overlap:
                animal_detected = True
                detected_species = species_name
                animal_conf = conf

                # Draw Red Box for Genuine Wildlife
                cv2.rectangle(frame, (wb[0], wb[1]), (wb[2], wb[3]), (0, 0, 255), 2)
                label = f"{detected_species} {conf:.2f}"
                cv2.putText(frame, label, (wb[0], wb[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 3. Alert Dispatch Logic
    current_time = time.time()
    if animal_detected and (current_time - last_alert_time > COOLDOWN_TIME):
        print(f"\n[ALARM] Genuine {detected_species} detected!")
        print(">> Triggering Highway Crossing Alert...")

        payload = {
            "node_id": "Highway-Node-001",
            "species": detected_species,
            "confidence": round(animal_conf, 2),
            "latitude": 19.231,
            "longitude": 72.825,
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }
        try:
            requests.post("http://localhost:8000/api/alerts", json=payload, timeout=2)
            print(f">> Payload successfully transmitted to Cloud Database!")
        except Exception as e:
            print(f">> [WARNING] Could not reach Cloud Backend: {e}")

        # Save latest annotated frame for the Streamlit dashboard
    cv2.imwrite(
        LATEST_FRAME_PATH,
        frame
    )

    # Show live Edge AI feed
    cv2.imshow(
        "Wildlife Edge AI - Live Feed",
        frame
    )

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources after loop
cap.release()
cv2.destroyAllWindows()

print("[INFO] System shutdown complete.")