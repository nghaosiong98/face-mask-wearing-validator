import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

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

def face_detection(frame, net, model):
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
		net.setInput(blob)
		detections = net.forward()
		for i in range(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]
				if confidence > args['confidence']:
						box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
						(startX, startY, endX, endY) = box.astype("int")

						(startX, startY) = (max(0, startX - 10), max(0, startY - 10))
						(endX, endY) = (min(w - 1, endX + 10), min(h - 1, endY + 10))

						text = "{:.2f}%".format(confidence * 100)
						y = startY - 10 if startY - 10 > 10 else startY + 10
						# cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
						# cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

						face = frame[startY:endY, startX:endX]
						# face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
						face = cv2.resize(face, (224, 224))
						# face = img_to_array(face)
						face = preprocess_input(face)
						face = np.expand_dims(face, axis=0)

						(correct, incorrect) = model.predict(face)[0]

						label = "Correct" if correct > incorrect else "Incorrect"
						color = (0, 255, 0) if label == "Correct" else (0, 0, 255)

						label = "{}: {:.2f}%".format(label, max(correct, incorrect) * 100)

						cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
						cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


def main():
		cap = cv2.VideoCapture(0)
		print("[INFO] loading face detector model...")
		prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt.txt"])
		weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
		faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

		print("[INFO] loading face mask wearing classification model...")
		mobilenetv2_model = load_model(args['model'])

		while(True):
				ret, frame = cap.read()

				# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				detections = face_detection(frame, faceNet, mobilenetv2_model)

				cv2.imshow('frame',frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
						break

		cap.release()
		cv2.destroyAllWindows()

if __name__ == "__main__":
		main()
