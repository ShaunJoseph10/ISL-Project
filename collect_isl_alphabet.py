import cv2
import mediapipe as mp
import numpy as np
import os

letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
samples_per_letter = 10

if not os.path.exists('data'):
    os.makedirs('data')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

current_letter_index = 0
sample_count = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
) as hands:

    print("Press S to save sample")
    print("Press Q to quit")

    while current_letter_index < len(letters):

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        current_letter = letters[current_letter_index]
        height = frame.shape[0]

        landmarks = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

        # Bottom-left text
        cv2.putText(frame, f"Letter: {current_letter}",
                    (10, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.putText(frame, f"Samples: {sample_count}/{samples_per_letter}",
                    (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2)

        cv2.imshow("ISL Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and landmarks is not None:

            letter_path = f"data/{current_letter}"
            if not os.path.exists(letter_path):
                os.makedirs(letter_path)

            np.save(f"{letter_path}/sample_{sample_count}.npy", landmarks)
            print(f"Saved {current_letter} sample {sample_count}")
            sample_count += 1

        if sample_count >= samples_per_letter:
            print(f"Completed {current_letter}")
            current_letter_index += 1
            sample_count = 0

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
