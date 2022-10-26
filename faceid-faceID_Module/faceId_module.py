from imutils.video import VideoStream
import face_recognition as fr
import imutils
import time
import pickle
import cv2
from threading import Thread, Event
from typing import Callable


class FaceId(Thread):

    def __init__(
        self,
        output_array: list = None,
        callback: Callable = None,
        return_bounding_boxes=False,
        encoding_file='encodings.pickle',
        webcam_src=0,
        usePiCamera=True,
        debug=False
    ):
        super(FaceId, self).__init__(group=None, target=None, name=None)
        self.vs = None
        self.encodings = None
        self._stop_event = Event()
        self.output_array = output_array
        self.callback = callback
        self.return_bounding_boxes = return_bounding_boxes
        self.encoding_file = encoding_file
        self.debug = debug
        self.webcam_src = webcam_src
        self.model = "hog"
        self.usePiCamera = usePiCamera

    def start_recognition(self):

        self.encodings = self.load_encoding()
        self.vs, _ = self.open_webcam()

        self.start()

    def run(self):

        while not self._stop_event.is_set():

            frame = self.vs.read()

            rgbFrame = self.process_frame(frame)

            names, boxes = self.recognize(rgbFrame)

            if self.output_array is not None and len(self.output_array) > 0:
                self.output_array[0] = (names, boxes)

            if self.callback is not None:
                self.callback(names, boxes)

    def end_recognition(self):
        self._stop_event.set()
        self.vs.stop()

    def is_running(self):
        return not self._stop_event.is_set()

    def load_encoding(self):

        self.log("[INFO] loading encodings...")

        with open(self.encoding_file, "rb") as f:
            data = f.read()

        return pickle.loads(data)

    def open_webcam(self):
        vs = VideoStream(self.webcam_src, usePiCamera=self.usePiCamera).start()

        writer = None

        # Make sure the camera is up and running
        time.sleep(2.0)

        return vs, writer

    def process_frame(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)

        return rgb

    def recognize(self, videoFrame):

        self.log("[INFO] recognizing faces...")

        boxes = fr.face_locations(videoFrame, model=self.model)
        frameEncoding = fr.face_encodings(videoFrame, boxes)

        names = []

        for encoding in frameEncoding:

            matches = fr.compare_faces(self.encodings['encodings'], encoding)
            name = "Unknown"

            if True in matches:

                matchedIdxs = [i for (i, b) in enumerate(matches) if b]

                counts = {}

                for i in matchedIdxs:
                    name = self.encodings["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                name = max(counts, key=counts.get)

            names.append(name)

        return (names, boxes)

    def log(self, message):
        if self.debug:
            print(message)


def print_output(names, boxes):
    print('Name %s, Boxes: %s' % (names, boxes))


if __name__ == '__main__':
    faceid = FaceId(callback=print_output, usePiCamera=False)

    faceid.start_recognition()

    time.sleep(5)

    faceid.end_recognition()
