import cv2
import time
import numpy as np
import PoseModule as pm

cap = cv2.VideoCapture("AITrainer/curling2.mp4")
pTime = 0
detector = pm.poseDetector()
count = 0
direc = 0

while True:
    success, img = cap.read()
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    # print(lmList)
    if len(lmList) != 0:
        # right arm
        # detector.findAngle(img, 12, 14, 16)
        # left arm
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (190, 310), (0, 100))
        # print(per)

        # check for the number of curls
        if per == 100:
            if direc == 0:
                count += 0.5
                direc = 1
        if per == 0:
            if direc == 1:
                count += 0.5
                direc = 0
        # print(count)
        cv2.putText(img, f'Curls: {int(count)}', (50, 150),
                    cv2.FONT_HERSHEY_PLAIN, 3, (240, 180, 10), 3)

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (50, 100), cv2.FONT_HERSHEY_PLAIN,
                2, (0, 255, 200), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
