from ultralytics import YOLO

def main():
    # Load pre-trained Nano weights for fine-tuning
    model = YOLO('yolov8n.pt')

    # Train on the cleaned 16-class dataset
    results = model.train(
        data='./dataset_cleaned/wildlife_16.yaml',
        epochs=30,
        imgsz=640,
        batch=16,          
        device=0,          
        workers=2,         
        project='SIH_Wildlife',
        name='clean_wildlife_model'
    )

# This is required for Windows multiprocessing
if __name__ == '__main__':
    # freeze_support() is sometimes needed by PyTorch on Windows
    from multiprocessing import freeze_support
    freeze_support()
    
    main()