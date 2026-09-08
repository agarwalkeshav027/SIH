import os
from ultralytics import YOLO
import torch
torch.set_num_threads(8)  # Set to the physical core count

# =====================================================================
# 1. Map paths to the cleaned dataset YAML and existing model weights
# =====================================================================
model_path = os.path.abspath("SIH_Wildlife/edge_prototype/weights/best.pt")
yaml_path = os.path.abspath("dataset_cleaned/wildlife_16.yaml")

print(f"[INFO] Loading model from: {model_path}")
print(f"[INFO] Using dataset config: {yaml_path}")

# Verify paths exist before executing validation
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model weights not found at: {model_path}")

if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"Dataset YAML not found at: {yaml_path}. Run the cleaning script first.")

# =====================================================================
# 2. Load the fine-tuned YOLOv8 model
# =====================================================================
model = YOLO(model_path)

# =====================================================================
# 3. Run validation sequence on the cleaned TEST split
# =====================================================================
print("\n[INFO] Starting evaluation on the cleaned 16-class TEST split...")

metrics = model.val(
    data=yaml_path,
    split='test',
    imgsz=640,
    batch=16,
    device='cpu',        # Change to 0 or 'cuda' if running with an NVIDIA GPU
    project='SIH_Wildlife',
    name='edge_test_cleaned_results',
    exist_ok=True        # Overwrite/reuse the run directory without creating edge_test_cleaned_results2, etc.
)

# =====================================================================
# 4. Output evaluation metrics for your pitch/presentation
# =====================================================================
print("\n" + "=" * 55)
print("🎯 EVALUATION COMPLETE (CLEANED 16-CLASS TEST SET):")
print("=" * 55)
print(f"Mean Average Precision @ 0.50 (mAP50)    : {metrics.box.map50 * 100:.2f}%")
print(f"Mean Average Precision @ 0.50:0.95 (mAP) : {metrics.box.map * 100:.2f}%")
print(f"Overall Precision                       : {metrics.box.mp * 100:.2f}%")
print(f"Overall Recall                          : {metrics.box.mr * 100:.2f}%")
print("=" * 55)