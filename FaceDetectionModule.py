import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionCon=0.5):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            self.minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)

        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic, = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img,
                                f'Accurary: {int(detection.score[0]*100)}%',
                                (bbox[0], bbox[1]-15), cv2.FONT_HERSHEY_DUPLEX,
                                0.5, (128, 242, 69), 1)

        return img, bboxs

    def fancyDraw(self, img, bbox, ln=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1, = x + w, y + h
        cv2.rectangle(img, bbox, (153, 255, 20), rt)
        # top left x, y
        cv2.line(img, (x, y), (x + ln, y), (153, 255, 20), t)
        cv2.line(img, (x, y), (x, y + ln), (153, 255, 20), t)
        # top right x1, y
        cv2.line(img, (x1, y), (x1 - ln, y), (153, 255, 20), t)
        cv2.line(img, (x1, y), (x1, y + ln), (153, 255, 20), t)
        # bottom left x, y1
        cv2.line(img, (x, y1), (x + ln, y1), (153, 255, 20), t)
        cv2.line(img, (x, y1), (x, y1 - ln), (153, 255, 20), t)
        # bottom right x1, y1
        cv2.line(img, (x1, y1), (x1 - ln, y1), (153, 255, 20), t)
        cv2.line(img, (x1, y1), (x1, y1 - ln), (153, 255, 20), t)
        return img


def main():
    cap = cv2.VideoCapture("Poses/girlface.mp4")
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        print(bboxs)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'Fps: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_DUPLEX,
                    1, (128, 242, 69), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
