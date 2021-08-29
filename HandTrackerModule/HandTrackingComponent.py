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
        
        cap.set(3,1280)
        cap.set(4,720)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        while True:
            success,img = cap.read()

            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img,handLms,self.mp_hands.HAND_CONNECTIONS)
            img = cv2.flip(img,2)
            cv2.imshow("Tracker",img)
            cv2.waitKey(1)

    def moveObject(self, cameraId=0,handNumber=1):

        cap = cv2.VideoCapture(cameraId)
        cap.set(3,1280)
        cap.set(4,720)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        # Create object
        # this can be made dynamic
        cx,cy,r = 100,100,50
        color = (255,0,0)

        while True:
            success,img = cap.read()

            image_height, image_width, _ = img.shape
            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                
                for handLms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img,handLms,self.mp_hands.HAND_CONNECTIONS)

                    for id,landmark in enumerate(handLms.landmark) :
                        if id == 8:
                            pt1,pt2 = int(landmark.x * image_width), int(landmark.y * image_height)
                            if cx-r<pt1<cx+r :
                                cx=pt1
                                cy=pt2
                                color = (0,255,0)
                            else:
                                color = (255,0,0)
            
            cv2.circle(img,(cx,cy),r,color,-1)
                    
            img = cv2.flip(img,2)
            cv2.imshow("Tracker",img)
            cv2.waitKey(1)

def main():
    handTrackerMod = handTracker(detectionConfidence=0.7,trackConfidence=0.7)
    #handTrackerMod.startHandTracking(0,1)
    handTrackerMod.moveObject()

if __name__ == "__main__":
    main()




    
    
        








