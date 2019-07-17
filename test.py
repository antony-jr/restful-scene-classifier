from sys import argv,exit
from RestFulSceneClassifier import Places365Classifier


if len(argv) == 1:
    exit(0)

img_path = str(argv[1])
places = Places365Classifier.Places365Classifier()

places.digest(img_path , base64Encoded = False)
