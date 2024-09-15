
# Import necessary libraries
from ultralytics import YOLO
import cv2
import numpy as np


# Load the YOLOv8 model
model = YOLO ("best.pt")  # Ensure 'best.pt' is in the correct directory
class_names = model.names  # Get class names from the model

# Open video file
cap = cv2.VideoCapture("E:\Project\yolov8-roadpothole-detection-main\p.mp4")  # Ensure 'p.mp4' is in the correct directory
count = 0

while True:
    ret, img = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit the loop if there are no more frames
    count += 1
    if count % 3 != 0:
        continue  # Process every 3rd frame

    img = cv2.resize(img, (1020, 500))  # Resize the image
    h, w, _ = img.shape  # Get image dimensions
    results = model.predict(img)  # Perform inference

    for r in results:
        boxes = r.boxes  # Boxes object for bounding box outputs
        masks = r.masks  # Masks object for segmentation masks outputs
        
        if masks is not None:
            masks = masks.data.cpu().numpy()  # Move masks to CPU and convert to numpy array
            for seg, box in zip(masks, boxes):
                seg = cv2.resize(seg, (w, h))  # Resize mask to match the original image size
                contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    d = int(box.cls)  # Get class index
                    c = class_names[d]  # Get class name
                    x, y, x1, y1 = cv2.boundingRect(contour)  # Get bounding box coordinates
                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)  # Draw contour
                    cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Put class name

    cv2.imshow('img', img)  # Show the processed image
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows