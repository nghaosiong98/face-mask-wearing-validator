from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os


class MaskValidator:
    def __init__(self):
        self.model = load_model('mask_classification_model')
        prototxt_path = os.path.sep.join(['face_detector', "deploy.prototxt.txt"])
        weight_path = os.path.sep.join(['face_detector', "res10_300x300_ssd_iter_140000.caffemodel"])
        self.face_net = cv2.dnn.readNet(prototxt_path, weight_path)

    # This function resize the input image
    def preprocess_image(self, image):
        preprocessed_image = cv2.resize(image, (224, 224))
        preprocessed_image = preprocess_input(preprocessed_image)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        return preprocessed_image

    # Perform prediction on the image and return label based on the
    def predict(self, face_image):
        preprocessed_image = self.preprocess_image(face_image)
        (correct, incorrect) = self.model.predict(preprocessed_image)[0]
        label = "Correct" if correct > incorrect else "Incorrect"
        return label

    # Perform mask wearing validation
    # Step 1: detect and locate face
    # Step 2: predict correctness on the face image
    def validate_mask(self, image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(
            image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX - 10), max(0, startY - 10))
                (endX, endY) = (min(w - 1, endX + 10), min(h - 1, endY + 10))

                face_image = image[startY:endY, startX:endX]
                label = self.predict(face_image)

                return {
                    'startX': int(startX),
                    'startY': int(startY),
                    'endX': int(endX),
                    'endY': int(endY),
                    'label': str(label),
                }
        return None

    # def predict(self, image_string):
    #     nparr = np.fromstring(image_string, np.uint8)
    #     raw_image = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    #     startX, startY, endX, endY = self.detect_face(raw_image)
    #     face_image = raw_image[startY:endY, startX:endX]
    #     preprocessed_image = self.preprocess_image(face_image)
    #     label = self.validate_mask(preprocessed_image)
    #     return {
    #         'startX': int(startX),
    #         'startY': int(startY),
    #         'endX': int(endX),
    #         'endY': int(endY),
    #         'label': str(label),
    #     }
