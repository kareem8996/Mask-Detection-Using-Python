import tensorflow as tf
import numpy as np
from imutils.video import VideoStream
import imutils
import time
import os
import serial
import time 
import cv2

#RGB
GREEN = (0,255,0) 
RED = (0,0,255)

lowConfidence = 0.75

#face detectinon function
def DetectandPredict(frame, faceNet, maskNet):

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	preds = []
	faces = []
	locs = []

	for i in range(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is greater than the minimum confidence
		if confidence > lowConfidence:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel ordering, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = tf.keras.preprocessing.image.img_to_array(face)
			face = tf.keras.applications.mobilenet_v2.preprocess_input(face)

			
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	return (locs, preds)

# load our pre trained face detector model gathered from the internet
prototxtPath = r"C:\Users\karee\Desktop\python projects\Mask detect\deploy_prototxt.py"																		
weightsPath = r"C:\Users\karee\Desktop\python projects\Mask detect\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = tf.keras.models.load_model(r"C:\Users\karee\Desktop\python projects\Mask detect\mask_detector.model")
	
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	
	frame = vs.read()
	frame = imutils.resize(frame, width=900)

	(locs, preds) = DetectandPredict(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding locations
	for (box, pred) in zip(locs, preds):


		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		if label =="Mask":
			print("GRANTED")
			
		else: 
			print("DENIED")
			
		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	
	cv2.imshow("FaceMask Detection by KAREEM -- q to quit", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()