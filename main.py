import cv2
import socket
import json
import time
from ultralytics import YOLO

# -----------------------------
# UDP Setup (ESP32 communication)
# -----------------------------
UDP_IP = "127.0.0.1"      # change later to ESP32 IP
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -----------------------------
# Load AI Model
# -----------------------------
model = YOLO("best.pt")

# -----------------------------
# Camera Stream (Phone)
# -----------------------------
STREAM_URL = "http://10.47.188.99:8080/video"

cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("ERROR: Could not open phone camera stream")
    exit()
else:
    print("Phone camera connected successfully")

# reduce buffering lag
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# -----------------------------
# Detection Settings
# -----------------------------
confidence_threshold = 0.6
min_box_area = 5000

# -----------------------------
# Command Rate Limiter
# -----------------------------
last_command_time = 0
command_interval = 0.4

# -----------------------------
# AI State System
# -----------------------------
search_direction = "LEFT"
last_search_switch = time.time()

# -----------------------------
# Send command to boat
# -----------------------------
def send_command(move, speed):

    global last_command_time

    if time.time() - last_command_time < command_interval:
        return

    packet = {
        "move": move,
        "speed": speed
    }

    message = json.dumps(packet)

    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

    print("Sent:", message)

    last_command_time = time.time()

# -----------------------------
# Main Loop
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        break

    # reduce resolution for faster processing
    frame = cv2.resize(frame, (640, 480))

    frame_height, frame_width = frame.shape[:2]

    # AI Detection
    results = model(frame, conf=confidence_threshold)

    plastic_found = False
    target_x = None

    for r in results:

        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            area = (x2 - x1) * (y2 - y1)

            if area < min_box_area:
                continue

            if label == "plastic":

                plastic_found = True
                target_x = (x1 + x2) // 2

    # -----------------------------
    # Navigation Logic
    # -----------------------------
    if plastic_found:

        center = frame_width // 2
        offset = target_x - center

        # dead zone to prevent jitter
        if abs(offset) < 120:

            state = "FORWARD"
            send_command("forward", 70)

        elif offset < 0:

            state = "LEFT"
            send_command("left", 60)

        else:

            state = "RIGHT"
            send_command("right", 60)

    else:

        state = "SEARCH"

        # change search direction every 2 seconds
        if time.time() - last_search_switch > 2:

            if search_direction == "LEFT":
                search_direction = "RIGHT"
            else:
                search_direction = "LEFT"

            last_search_switch = time.time()

        if search_direction == "LEFT":
            send_command("left", 40)
        else:
            send_command("right", 40)

    # -----------------------------
    # Display frame
    # -----------------------------
    annotated_frame = results[0].plot()

    cv2.putText(
        annotated_frame,
        state,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Boat AI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()