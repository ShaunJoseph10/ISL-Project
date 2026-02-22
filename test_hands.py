import cv2
import mediapipe as mp
import numpy as np
import pickle

# =========================
# Load trained model
# =========================
with open("isl_model.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# MediaPipe Setup
# (MUST match collect_isl.py)
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,              # 🔥 IMPORTANT (same as collection)
    min_detection_confidence=0.8,    # 🔥 Same as collection
    min_tracking_confidence=0.8      # 🔥 Same as collection
)

# =========================
# Camera Setup
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press Q to quit")

# =========================
# Main Loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror view
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    prediction = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            landmarks = []

            # Use ONLY x and y (42 features)
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            landmarks = np.array(landmarks).reshape(1, -1)

            # Predict
            prediction = model.predict(landmarks)[0]

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # =========================
    # Purple Text (BGR: 255,0,255)
    # =========================
    cv2.putText(
        frame,
        f"Letter: {prediction}",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),  # 💜 Purple
        2
    )

    cv2.imshow("ISL Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()