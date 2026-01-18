import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

# -------------------------
# Config
# -------------------------
MODEL_PATH = "model/gesture_model.joblib"
GESTURE_LABELS = [
    "Open Palm",
    "Fist",
    "Thumbs Up",
    "Thumbs Down",
    "Peace"
]

PRED_WINDOW = 8
CONF_THRESHOLD = 0.6

# -------------------------
# Load model
# -------------------------
model = joblib.load(MODEL_PATH)

# -------------------------
# Prediction smoothing
# -------------------------
pred_queue = deque(maxlen=PRED_WINDOW)

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
# Webcam
# -------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# -------------------------
# Main loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_text = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract landmarks
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            features = np.array(features).reshape(1, -1)

            # Predict
            probs = model.predict_proba(features)[0]
            pred_queue.append(probs)

            avg_probs = np.mean(pred_queue, axis=0)
            confidence = np.max(avg_probs)
            gesture_id = np.argmax(avg_probs)

            if confidence > CONF_THRESHOLD:
                gesture_text = f"{GESTURE_LABELS[gesture_id]} ({confidence:.2f})"
            else:
                gesture_text = "Detecting..."

    cv2.putText(
        frame,
        gesture_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
