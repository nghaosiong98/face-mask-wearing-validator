from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import werkzeug
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_classification_model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float,
                default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


class Predict(Resource):
    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument("image", type=werkzeug.datastructures.FileStorage, location='files')
        self.req_parser = parser

        self.model = load_model(args['model'])
        prototxt_path = os.path.sep.join([args["face"], "deploy.prototxt.txt"])
        weight_path = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
        self.face_net = cv2.dnn.readNet(prototxt_path, weight_path)

    def detect_face(self, image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(
            image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > args['confidence']:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX - 10), max(0, startY - 10))
                (endX, endY) = (min(w - 1, endX + 10), min(h - 1, endY + 10))

                return startX, startY, endX, endY

    def mask_wearing_validate(self, image):
        preprocessed_image = cv2.resize(image, (224, 224))
        preprocessed_image = preprocess_input(preprocessed_image)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        (correct, incorrect) = self.model.predict(preprocessed_image)[0]
        label = "Correct" if correct > incorrect else "Incorrect"
        return label

    def post(self):
        image_file = self.req_parser.parse_args(strict=True).get("image", None)
        if image_file:
            imageString = image_file.read()
            nparr = np.fromstring(imageString, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
            startX, startY, endX, endY = self.detect_face(img)
            face_image = img[startY:endY, startX:endX]
            label = self.mask_wearing_validate(face_image)
            return jsonify({
                'startX': int(startX),
                'startY': int(startY),
                'endX': int(endX),
                'endY': int(endY),
                'label': str(label),
            })
        else:
            return "Failed"


app = Flask(__name__)
api = Api(app)

api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
