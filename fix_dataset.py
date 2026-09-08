import os
import shutil
import yaml

# 1. Define Old-to-New Class Mapping (54 classes -> 16 target classes)
REMAP_DICT = {
    0: (0, 'Zebra'),
    1: (1, 'Lion'),
    2: (2, 'Leopard'),
    3: (3, 'Cheetah'),
    4: (4, 'Tiger'),
    5: (5, 'Bear'),
    9: (6, 'Bull'),
    17: (7, 'Elephant'),
    18: (8, 'Horse'),
    19: (9, 'Fox'),
    22: (10, 'Kangaroo'),
    23: (11, 'Deer'),
    35: (12, 'Hippopotamus'),
    36: (13, 'BrownBear'),
    37: (14, 'Rhinoceros'),
    42: (15, 'Jaguar')
}

TARGET_OLD_IDS = set(REMAP_DICT.keys())

# Paths matching your exact directory layout
SOURCE_DIR = "./dataset"          # Contains dataset/test/images and dataset/test/labels
OUTPUT_DIR = "./dataset_cleaned"  # Clean output destination

SPLITS = ['test']

print("[INFO] Starting test dataset cleaning and class ID remapping...")

for split in SPLITS:
    # Matching: dataset/test/images and dataset/test/labels
    src_img_dir = os.path.join(SOURCE_DIR, split, 'images')
    src_lbl_dir = os.path.join(SOURCE_DIR, split, 'labels')
    
    # Target output structure: dataset_cleaned/test/images and dataset_cleaned/test/labels
    dst_img_dir = os.path.join(OUTPUT_DIR, split, 'images')
    dst_lbl_dir = os.path.join(OUTPUT_DIR, split, 'labels')
    
    if not os.path.exists(src_lbl_dir):
        print(f"[ERROR] Source label directory not found at: {src_lbl_dir}")
        continue
        
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    
    kept_images = 0
    total_labels_remapped = 0
    
    for label_file in os.listdir(src_lbl_dir):
        if not label_file.endswith('.txt'):
            continue
            
        src_label_path = os.path.join(src_lbl_dir, label_file)
        valid_lines = []
        
        with open(src_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    old_cls_id = int(parts[0])
                except ValueError:
                    continue
                
                if old_cls_id in TARGET_OLD_IDS:
                    new_cls_id = REMAP_DICT[old_cls_id][0]
                    new_line = f"{new_cls_id} " + " ".join(parts[1:]) + "\n"
                    valid_lines.append(new_line)
                    total_labels_remapped += 1
        
        # Only copy image and label if at least one target class is present
        if valid_lines:
            base_name = os.path.splitext(label_file)[0]
            img_found = False
            
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']:
                img_name = base_name + ext
                src_img_path = os.path.join(src_img_dir, img_name)
                
                if os.path.exists(src_img_path):
                    shutil.copy(src_img_path, os.path.join(dst_img_dir, img_name))
                    img_found = True
                    break
            
            if img_found:
                dst_label_path = os.path.join(dst_lbl_dir, label_file)
                with open(dst_label_path, 'w') as f:
                    f.writelines(valid_lines)
                kept_images += 1
            else:
                print(f"[WARN] Label exists but image not found for: {base_name}")
                
    print(f"[{split.upper()}] Processed: Kept {kept_images} images with {total_labels_remapped} valid bounding boxes.")

# 2. Write the verified YAML config
class_names = [REMAP_DICT[old_id][1] for old_id in sorted(REMAP_DICT.keys(), key=lambda x: REMAP_DICT[x][0])]

yaml_data = {
    'path': os.path.abspath(OUTPUT_DIR),
    'val': 'test/images',     # Points to dataset_cleaned/test/images
    'test': 'test/images',
    'nc': len(class_names),
    'names': class_names
}

yaml_path = os.path.join(OUTPUT_DIR, 'wildlife_16.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print(f"\n[SUCCESS] Clean dataset ready at: {os.path.abspath(OUTPUT_DIR)}")
print(f"[SUCCESS] Dataset YAML generated at: {yaml_path}")