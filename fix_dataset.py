import os
import shutil
import yaml

# 1. Define Old-to-New Class Mapping
# Maps original data.yaml (54 classes) to clean filtered (16 classes)
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

# Paths (Adjust if your raw dataset folder is named differently)
SOURCE_DIR = "./dataset"          # Folder containing images/ and labels/
OUTPUT_DIR = "./dataset_cleaned"  # Clean output folder

SPLITS = ['train', 'valid', 'test']

print("[INFO] Starting dataset cleaning and index remapping...")

for split in SPLITS:
    src_img_dir = os.path.join(SOURCE_DIR, 'images', split)
    src_lbl_dir = os.path.join(SOURCE_DIR, 'labels', split)
    
    dst_img_dir = os.path.join(OUTPUT_DIR, 'images', split)
    dst_lbl_dir = os.path.join(OUTPUT_DIR, 'labels', split)
    
    if not os.path.exists(src_lbl_dir):
        print(f"[SKIP] Split '{split}' not found at {src_lbl_dir}")
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
                old_cls_id = int(parts[0])
                
                if old_cls_id in TARGET_OLD_IDS:
                    new_cls_id = REMAP_DICT[old_cls_id][0]
                    # Rewrite line with the new mapped ID
                    new_line = f"{new_cls_id} " + " ".join(parts[1:]) + "\n"
                    valid_lines.append(new_line)
                    total_labels_remapped += 1
        
        # Only copy image and label if at least one target animal is present
        if valid_lines:
            base_name = os.path.splitext(label_file)[0]
            # Check image extensions
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                img_name = base_name + ext
                src_img_path = os.path.join(src_img_dir, img_name)
                if os.path.exists(src_img_path):
                    shutil.copy(src_img_path, os.path.join(dst_img_dir, img_name))
                    break
            
            dst_label_path = os.path.join(dst_lbl_dir, label_file)
            with open(dst_label_path, 'w') as f:
                f.writelines(valid_lines)
                
            kept_images += 1
            
    print(f"[{split.upper()}] Processed: Kept {kept_images} images with {total_labels_remapped} valid bounding boxes.")

# 2. Write the verified YAML config
yaml_data = {
    'path': os.path.abspath(OUTPUT_DIR),
    'train': 'images/train',
    'val': 'images/valid',
    'test': 'images/test' if os.path.exists(os.path.join(OUTPUT_DIR, 'images', 'test')) else 'images/valid',
    'nc': 16,
    'names': [REMAP_DICT[old_id][1] for old_id in sorted(REMAP_DICT.keys(), key=lambda x: REMAP_DICT[x][0])]
}

yaml_path = os.path.join(OUTPUT_DIR, 'wildlife_16.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print(f"\n[SUCCESS] Clean dataset ready at: {OUTPUT_DIR}")
print(f"[SUCCESS] Dataset YAML generated at: {yaml_path}")