import face_recognition as fr
import argparse
import pickle
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encodings", required=True,
                    help="path to serialized db of facial encodings")
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    ap.add_argument("-d", "--detection-method", type=str, default="hog",
                    help="face detection model to use: either `hog` or `cnn`")
    ap.add_argument("-s", "--save", type=int, default=0,
                    help="save the result instead of showing it")
    ap.add_argument("-n", "--name", type=str, default="Image",
                    help="The display name of the image")
    args = vars(ap.parse_args())

    encodings = load_encoding(args["encodings"])
    (image, original) = load_image(args["image"])

    (names, boxes) = recognize(image, args["detection_method"], encodings)

    draw_boxes(original, boxes, names, args["name"], args["save"])


def load_encoding(path):

    print("[INFO] loading encodings...")

    with open(path, "rb") as f:
        data = f.read()

    return pickle.loads(data)


def load_image(imagePath: str):
    image = cv2.imread(imagePath)

    return (cv2.cvtColor(image, cv2.COLOR_BGR2RGB), image)


def recognize(image, model, encodings):

    print("[INFO] recognizing faces...")

    boxes = fr.face_locations(image, model=model)
    imageEncoding = fr.face_encodings(image, boxes)

    names = []

    for encoding in imageEncoding:

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

    return (names, boxes)


def draw_boxes(image, boxes, names, imageName="Image", save=False):

    for ((top, right, bottom, left), name) in zip(boxes, names):

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        y = top - 15 if top - 15 > 15 else top + 15

        cv2.putText(image, name, (left, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    if save > 0:
        cv2.imwrite("output/%s.jpg" % (imageName,), image)
    else:
        cv2.imshow(imageName, image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
