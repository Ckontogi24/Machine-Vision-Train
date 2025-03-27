from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")  # YOLO11 medium μοντέλο για segmentation
model.train(data="/home/roboticslab/Documents/crack_segmentation_merged_dataset/data.yaml", epochs=300, imgsz=448, batch=2)
