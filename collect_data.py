import cv2
import mediapipe as mp
import numpy as np
import os

# -------------------------
# Config
# -------------------------
GESTURES = {
    0: "Open_Palm",
    1: "Fist",
    2: "Thumbs_Up",
    3: "Thumbs_Down",
    4: "Peace"
}

SAMPLES_PER_GESTURE = 200
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------
# MediaPipe setup
# -------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# -------------------------
# Data containers
# -------------------------
X, y = [], []

# -------------------------
# Webcam
# -------------------------
cap = cv2.VideoCapture(0)

print("Press keys 0–4 to collect gestures")
print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            key = cv2.waitKey(1) & 0xFF

            if key in [ord(str(k)) for k in GESTURES]:
                label = int(chr(key))
                X.append(landmarks)
                y.append(label)
                print(f"Collected {GESTURES[label]} : {len([v for v in y if v == label])}")

    cv2.imshow("Collect Gestures", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------
# Save dataset
# -------------------------
np.save(os.path.join(DATA_DIR, "X.npy"), np.array(X))
np.save(os.path.join(DATA_DIR, "y.npy"), np.array(y))

print("Dataset saved!")
