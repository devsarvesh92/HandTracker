import cv2
import mediapipe as mp
import time
import math

class finger(object):
    pass

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
        cx,cy,r = 100,100,100
        color = (255,0,0)
        success,img = cap.read()
        finger1 = finger()

        while True:
            success,img = cap.read()

            self.image_height, self.image_width, _ = img.shape
            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

            if results.multi_hand_landmarks:
                
                for handLms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img,handLms,self.mp_hands.HAND_CONNECTIONS)
                    self.handLms = handLms
                    distance = self.getDistance(8,12)
                    finger1.cx,finger1.cy = self.getCordinates(8)
                    if distance is not None and distance < 70:
                        if finger1.cx-r < cx < finger1.cx+r:
                            cx = finger1.cx
                            cy = finger1.cy
                            print(finger1)
           
            cv2.circle(img,(cx,cy),r,color,-1)

            img = cv2.flip(img,2)
            cv2.imshow("Tracker",img)
            cv2.waitKey(1)

    def getDistance(self,id1,id2):

        pt1 = finger()
        pt2 = finger()

        if hasattr(self,'handLms'):
            pt1.cx,pt1.cy = self.getCordinates(id1)
            pt2.cx,pt2.cy = self.getCordinates(id2)
            return math.dist([pt1.cx,pt1.cy],[pt2.cx,pt2.cy])

    def getCordinates(self,fingerId):
        for id,landmark in enumerate(self.handLms.landmark) :
            if id == fingerId:
                return (int(landmark.x * self.image_width), int(landmark.y * self.image_height))

def main():
    handTrackerMod = handTracker(detectionConfidence=0.7,trackConfidence=0.7)
    #handTrackerMod.startHandTracking(0,1)
    handTrackerMod.moveObject()

if __name__ == "__main__":
    main()




    
    
        








