from ultralytics import YOLO

# ==========================
# LOAD MODEL (only once)
# ==========================
model = YOLO("yolov8n.pt")   # pretrained nano model

# ==========================
# IMAGE PATH
# ==========================
image_path = r"C:\EDA_repo\AVSCODE\YOLO\img\download (1).jpg"

# ==========================
# PREDICTION + SAVE OUTPUT
# ==========================
results = model.predict(
    source=image_path,
    
    save=True,   # save output image
    
    project=r"C:\EDA_repo\AVSCODE\YOLO",  # main project folder
    name="runs",  # inside project -> runs/
    
    exist_ok=True,  # overwrite folder (no predict2, predict3)
    
    imgsz=320  # optional: faster inference
)

# ==========================
# PRINT RESULTS
# ==========================
print(results)