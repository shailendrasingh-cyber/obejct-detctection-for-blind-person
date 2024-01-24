import cv2
import pyttsx3

def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f]
    return net, classes

def get_objects(outputs, height, width, conf_threshold=0.5):
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x, center_y, w, h = map(int, detection[0:4] * [width, height, width, height])
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

def draw_labels(img, boxes, confidences, class_ids, classes):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    net, classes = load_yolo()
    cap = cv2.VideoCapture(0)

    engine = pyttsx3.init()

    while True:
        ret, frame = cap.read()
        height, width, _ = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        boxes, confidences, class_ids = get_objects(outs, height, width)
        draw_labels(frame, boxes, confidences, class_ids, classes)

        objects = set([classes[class_id] for class_id in class_ids])
        objects_description = ', '.join(objects)

        cv2.imshow("Object Detection", frame)
        engine.say(f"There are {len(objects)} objects: {objects_description} in front of you")
        engine.runAndWait()

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
