from ultralytics import YOLO
import cv2
import json
import socket

# -----------------------------
# UDP Communication Setup
# -----------------------------
UDP_IP = "127.0.0.1"   # For testing (local machine)
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -----------------------------
# Load Model
# -----------------------------
model = YOLO("best.pt")
print("Class names:", model.names)

cap = cv2.VideoCapture(0)

confidence_threshold = 0.5
last_state = "SEARCH"

# -----------------------------
# Command Generator
# -----------------------------
def generate_command(state):
    if state == "MOVE LEFT":
        return {"move": "left", "speed": 60}
    elif state == "MOVE RIGHT":
        return {"move": "right", "speed": 60}
    elif state == "MOVE FORWARD":
        return {"move": "forward", "speed": 70}
    elif state == "COLLECT":
        return {"move": "stop", "speed": 0}
    else:
        return {"move": "search", "speed": 40}

# -----------------------------
# Main Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    height, width, _ = frame.shape

    state = "SEARCH"

    for r in results:
        boxes = r.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Only detect plastic class (class 0)
                if cls != 0:
                    continue

                if conf < confidence_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                center_x = int((x1 + x2) / 2)
                box_width = int(x2 - x1)

                # Distance logic
                if box_width > width * 0.4:
                    state = "COLLECT"
                else:
                    if center_x < width / 3:
                        state = "MOVE LEFT"
                    elif center_x > 2 * width / 3:
                        state = "MOVE RIGHT"
                    else:
                        state = "MOVE FORWARD"

    # Send command only if changed
    if state != last_state:
        packet = generate_command(state)
        message = json.dumps(packet)

        sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

        print("Sent:", message)

        last_state = state

    # Show frame
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, state, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Boat AI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()