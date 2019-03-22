#--image = "images/example_01.jpg"
#--prototext = "MobileNetSSD_deploy.prototxt.txt"
#--model = "MobileNetSSD_deploy.caffemodel"


#Import necessary package
import numpy as np
import argparse
import cv2

#Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path to caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="Minimum probability to filter weak detection")

args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size = (len(CLASSES), 3))       #Bounding box COLORS

#Load the model
print("[INFO] loading the model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#Load our image
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]                #Extract height and width
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)           #Resizing and normalization parameters from mobilenets authors

#Pass the blob through network and obtain the detections and predictions
print("[INFO] computing object detection...")
net.setInput(blob)
detections = net.forward()

#Loop through the detection to determine what and where the objects arange

for i in np.arange(0,detections.shape[2]):
    #Extract the confidence associated with the prediction
    confidence = detections[0, 0, i, 2]

    #Filter out the weak detection by ensuring the confidence is graeter than minimum confidence

    if confidence > args["confidence"]:
        #Extract the index of the class label from the detections
        #Then compute the (x,y) coordinates of the bounding box for the object
        idx = int(detections[0, 0, i, 1])                           #Extract class labels
        box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])       #Bounding box around detected box
        (startX, startY, endX, endY) = box.astype("int")            #Extract x and y coordinate of the box for drawing the rectangle

        #Display the predictions
        labels = "{}:{:.2f}%".format(CLASSES[idx], confidence*100)
        print("[INFO] {}".format(labels))
        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, labels, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


#Show the image
cv2.imshow("Output", image)
cv2.waitKey(0)
