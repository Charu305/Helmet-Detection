from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load YOLO model
model = YOLO(r"C:\Users\ASUS\Documents\HOPE AI\Deep Learning\Week11-Deep Learning Module\Bike Helmet Detection\runs\detect\train\weights\best.pt")

# Image path
image_path = r"C:\Users\ASUS\Documents\HOPE AI\Deep Learning\Week11-Deep Learning Module\Bike Helmet Detection\HelmetDataset\test\images\BikesHelmets37_png.rf.d98f209c563814fb8c7a7aec37a548e7.jpg"

# Read image
image = cv2.imread(image_path)

# Check if loaded properly
if image is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

# Show original image
plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# YOLO prediction
results = model.predict(image, conf=0.30)

# Process detections
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # confidence filter
        if conf < 0.50:
            continue

        # area filter
        if (x2 - x1) * (y2 - y1) < 2000:
            continue

        label = f"{model.names[cls]}: {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# Show results
plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
