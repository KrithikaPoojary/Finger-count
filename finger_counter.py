import cv2
import mediapipe as mp

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Finger tip landmark IDs
tip_ids = [4, 8, 12, 16, 20]

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Flip for mirror effect
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    finger_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # Check which fingers are up
            if lm_list:
                # Thumb
                if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]:
                    finger_count += 1

                # Other fingers
                for id in range(1, 5):
                    if lm_list[tip_ids[id]][1] < lm_list[tip_ids[id] - 2][1]:
                        finger_count += 1

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show finger count
    cv2.rectangle(img, (0, 0), (200, 100), (0, 0, 0), -1)
    cv2.putText(img, f'Fingers: {finger_count}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Real-Time Finger Counter", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
    import cv2
import mediapipe as mp
import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7
)

# Tip IDs for fingers
tip_ids = [4, 8, 12, 16, 20]

# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Mirror view
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = img.shape

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            fingers = 0
            label = handedness.classification[0].label  # 'Left' or 'Right'

            if lm_list:
                # Thumb logic differs by hand
                if label == "Right":
                    if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]:
                        fingers += 1
                else:  # Left hand
                    if lm_list[tip_ids[0]][0] < lm_list[tip_ids[0] - 1][0]:
                        fingers += 1

                # Other 4 fingers (same logic for both hands)
                for id in range(1, 5):
                    if lm_list[tip_ids[id]][1] < lm_list[tip_ids[id] - 2][1]:
                        fingers += 1

            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show finger count per hand
            cv2.putText(img, f'{label} Hand: {fingers}', (10, 70 + idx * 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3)

    cv2.imshow("Finger Counter - Both Hands", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7
)

# Finger tip landmark IDs
tip_ids = [4, 8, 12, 16, 20]

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Flip image for mirror view
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    finger_counts = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape


cap.release()
cv2.destroyAllWindows()
