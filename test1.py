import cv2
import pyttsx3

text_speech = pyttsx3.init()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = []
classFile = "coco.names"
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'yolov3.cfg'
weightsPath = 'yolov3.weights'

net = cv2.dnn_DetectionModel(weightsPath, configPath)

net.setInputSize(416, 416)  # Adjust according to YOLO model requirements
net.setInputScale(1.0 / 255)  # YOLO typically uses values in the range [0, 1]
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Adjust the speech rate dynamically
            newVoiceRate = 40
            text_speech.setProperty('rate', newVoiceRate)
            answer = classNames[classId - 1]
            text_speech.say(answer)
            text_speech.runAndWait()

    cv2.imshow("output", img)
    cv2.waitKey(1)
