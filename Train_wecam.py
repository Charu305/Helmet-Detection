from ultralytics import YOLO
import cv2

# -------------------------
# 1. LOAD MODEL
# -------------------------
model = YOLO(
    r"C:\Users\ASUS\Documents\HOPE AI\Deep Learning\Week11-Deep Learning Module\Bike Helmet Detection\runs\detect\train\weights\best.pt"
)

print("Model Classes:", model.names)

COLOR_HELMET = (0, 255, 0)
COLOR_NO_HELMET = (0, 0, 255)

# -------------------------
# 2. START WEBCAM
# -------------------------
cap = cv2.VideoCapture(0)

# Reduce webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera not working")
    exit()

# -------------------------
# 3. DETECTION LOOP
# -------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Optional extra resize (for faster processing)
    frame = cv2.resize(frame, (640, 480))

    # YOLO prediction
    results = model(frame, conf=0.05, imgsz=320)

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{model.names[cls]} {conf:.2f}"

            if cls == 0:
                color = COLOR_HELMET
            else:
                color = COLOR_NO_HELMET

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()