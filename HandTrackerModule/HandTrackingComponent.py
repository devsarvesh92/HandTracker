import cv2
import mediapipe as mp
import time


class handTracker:
    def __init__(self, mode=False,maxHands = 2, detectionConfidence = 0.5, trackConfidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        self.mp_draw = mp.solutions.drawing_utils

    def startHandTracking(self, cameraId=0,handNumber=1):
        cap = cv2.VideoCapture(cameraId)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        while True:
            success,img = cap.read()
            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img,handLms,self.mp_hands.HAND_CONNECTIONS)
            cv2.imshow("Tracker",img)
            cv2.waitKey(1)

def main():
    handTrackerMod = handTracker()
    handTrackerMod.startHandTracking(0,1)

if __name__ == "__main__":
    main()




    
    
        








