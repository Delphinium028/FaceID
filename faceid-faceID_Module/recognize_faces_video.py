from imutils.video import VideoStream
import face_recognition as fr
import argparse
import imutils
import time
import pickle
import cv2
from threading import Thread


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encodings", required=True,
                    help="path to serialized db of facial encodings")
    ap.add_argument("-o", "--output", type=str,
                    help="path to output video")
    ap.add_argument("-y", "--display", type=int, default=1,
                    help="whether or not to display output frame to screen")
    ap.add_argument("-s", "--save", type=int, default=1,
                    help="whether or not to the out output frame to a file")
    ap.add_argument("-d", "--detection-method", type=str, default="hog",
                    help="face detection model to use: either `hog` or `cnn`")
    args = vars(ap.parse_args())

    encodings = load_encoding(args["encodings"])

    vs, writer = open_webcam()

    thread = None
    thread_boxes = [None]
    thread_names = [None]

    boxes = []
    names = []

    while True:

        frame = vs.read()

        rgbFrame, r = process_frame(frame)

        if thread is None:
            print("running detection")
            thread = Thread(target=recognize, args=(
                rgbFrame, args["detection_method"], encodings, thread_boxes, thread_names
            ))
            thread.start()
        elif not thread.is_alive() and thread_boxes[0] is not None and thread_names[0] is not None:

            boxes = thread_boxes[0]
            names = thread_names[0]

            thread = None
            thread_boxes = [None]
            thread_names = [None]

        # call if you don't want threading
        # (names, boxes) = recognize(rgbFrame,
        #                            args["detection_method"], encodings)

        draw_frame = draw_boxes(frame, r, boxes, names)

        if (args["save"] > 0):
            writer = write_frame(draw_frame, writer, args["output"])

        if args["display"] > 0:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    vs.stop()

    if writer is not None:
        writer.release()


def load_encoding(path):

    print("[INFO] loading encodings...")

    with open(path, "rb") as f:
        data = f.read()

    return pickle.loads(data)


def open_webcam(src=0):
    vs = VideoStream(src).start()

    writer = None

    time.sleep(2.0)

    return vs, writer


def process_frame(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)

    r = frame.shape[1] / float(rgb.shape[1])

    return rgb, r


def recognize(videoFrame, model, encodings, thread_boxes=None, thread_names=None):

    print("[INFO] recognizing faces...")

    boxes = fr.face_locations(videoFrame, model=model)
    frameEncoding = fr.face_encodings(videoFrame, boxes)

    names = []

    for encoding in frameEncoding:

        matches = fr.compare_faces(encodings['encodings'], encoding)
        name = "Unknown"

        if True in matches:

            matchedIdxs = [i for (i, b) in enumerate(matches) if b]

            counts = {}

            for i in matchedIdxs:
                name = encodings["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    if thread_names is not None and thread_boxes is not None:
        thread_names[0] = names
        thread_boxes[0] = boxes
    else:
        return (names, boxes)


def draw_boxes(frame, r, boxes, names):

    for ((top, right, bottom, left), name) in zip(boxes, names):

        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        y = top - 15 if top - 15 > 15 else top + 15

        cv2.putText(frame, name, (left, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return frame


def write_frame(frame, writer, output):

    if writer is None and output is not None:

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        writer = cv2.VideoWriter(
            output, fourcc, 20, (frame.shape[1], frame.shape[0]), True)

    if writer is not None:
        writer.write(frame)

    return writer


if __name__ == '__main__':
    main()
