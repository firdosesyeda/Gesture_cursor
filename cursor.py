import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize camera and Mediapipe Hands
cap = cv2.VideoCapture(0)
hand = mp.solutions.hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hand.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            draw.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # Get coordinates of index fingertip
            index_x = int(landmarks[8].x * w)
            index_y = int(landmarks[8].y * h)
            screen_x = int(landmarks[8].x * screen_w)
            screen_y = int(landmarks[8].y * screen_h)

            pyautogui.moveTo(screen_x, screen_y)

            # Get coordinates of thumb tip
            thumb_x = int(landmarks[4].x * w)
            thumb_y = int(landmarks[4].y * h)

            # Calculate Euclidean distance
            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)

            if distance < 40:
                pyautogui.click()

    cv2.imshow("Gesture Mouse", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
