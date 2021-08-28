import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

while True:
    success,img = cap.read()
    print(success)

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)


    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img,handLms,mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Tracker",img)
    cv2.waitKey(1)