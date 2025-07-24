import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import pickle

# Load the trained model
model = load_model("asl_sign_model.h5")

# Load labels exactly as saved during training
with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

img_size = 64  # input image size for your CNN model

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min = w, h
            x_max = y_max = 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            pad = 20
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w, x_max + pad)
            y_max = min(h, y_max + pad)

            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size == 0:
                continue

            roi_resized = cv2.resize(roi, (img_size, img_size))
            roi_normalized = roi_resized.astype('float32') / 255.0
            roi_input = np.expand_dims(roi_normalized, axis=0)

            prediction = model.predict(roi_input)
            predicted_index = np.argmax(prediction)
            predicted_letter = labels[predicted_index]
            confidence = prediction[0][predicted_index]

            # Show bounding box and prediction if confidence > 60%
            if confidence > 0.6:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{predicted_letter} ({confidence*100:.1f}%)",
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ASL Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
