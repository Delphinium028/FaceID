from imutils import paths
import face_recognition as fr
import argparse
import pickle
import cv2
import os


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", required=True,
                    help="path to input directory of faces + images")
    ap.add_argument("-e", "--encodings", required=True,
                    help="path to serialized db of facial encodings")
    ap.add_argument("-d", "--detection-method", type=str, default="hog",
                    help="face detection model to use: either `hog` or `cnn`")
    args = vars(ap.parse_args())

    imagePaths = list(paths.list_images(args["dataset"]))

    knownEncodings = []
    knownNames = []

    images = load_images(imagePaths)

    encode_images(images, args["detection_method"], knownEncodings, knownNames)

    save_encodings(args["encodings"], knownEncodings, knownNames)


def load_images(imagePaths: list) -> list:

    rbgImages = []

    for (i, imagePath) in enumerate(imagePaths):

        print("[INFO] loading image {}/{}".format(i + 1, len(imagePaths)))

        name = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        rbgImages.append((name, rgb))

    return rbgImages


def encode_images(rgbImages: tuple, detectionMethod: str, knownEncodings: list, knownNames: list):

    for (i, (name, image)) in enumerate(rgbImages):
        print("[INFO] encoding image {}/{}".format(i + 1, len(rgbImages)))
        boxes = fr.face_locations(image, model=detectionMethod)

        encodings = fr.face_encodings(image, boxes)

        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)


def save_encodings(encodingsPath, knownEncodings, knownNames):

    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    with open(encodingsPath, "wb") as f:
        f.write(pickle.dumps(data))


if __name__ == '__main__':
    main()
