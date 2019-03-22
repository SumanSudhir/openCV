#Import necessary library
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2


#Parshing of the argument
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to prototxt file")
ap.add_argument("-m", "--model", required=True, help="Path to cafe pre-trained model")
ap.add_argument("-c", "--confidence", type = float, default=0.2, help="minimum probability to filter weak detection")
args = vars(ap.parse_args())

#initiliaze the list of class labels mobilenets SSD was trained to detect and generate the bounding box for each class

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES),3))

#Load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#initiliaze the VideoStream, allow the camera sensor to warmup
print("[INFO] starting video stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)					#Wait for camera to warmup
fps = FPS().start()      #Start frame per second counting

#Loop over the each and every frames from the Video stream
while True:
	#Grab the frame from the video stream and resize it
	#To have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	#Grab the frame dimension and convert it to blob
	(h, w) = frame.shape[:2]
	#print(frame.shape)
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 0.007843, (300, 300), 127.5)

	#Pass the blob through the network and abtain the detections and prediction
	net.setInput(blob)
	detections = net.forward()
	#print(detections.shape)

	#We have now detected the object now we have to look at the confidence
	for i in np.arange(0, detections.shape[2]):
		#extract the confidence associated with predictions
		confidence = detections[0, 0, i, 2]
		#print(confidence)

		#Filter out the weaker detections
		if confidence > args["confidence"]:
			#Extract the index of the class labels from the detcions
			#Compute the (x, y) coordinates of the bounding box of the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")        #Compute the bounding box around the object

			#draw thw prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx] , confidence*100)
			cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY -15 > 15 else startY +15

			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	#Show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	#if the 'q ' key is pressed , break the loop
	if key == ord("q"):
		break

	#Update the FPS counter
	fps.update()

#stop the timer and display the FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

#Clean up
cv2.destroyAllWindows()
vs.stop()
