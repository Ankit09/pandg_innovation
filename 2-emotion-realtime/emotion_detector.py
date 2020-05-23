# USAGE
# python emotion_detector.py --cascade haarcascade_frontalface_default.xml --model checkpoints/epoch_75.hdf5

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained emotion detector CNN")
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
args = vars(ap.parse_args())

# load the face detector cascade, emotion detection CNN, then define
# the list of emotion labels
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])
EMOTIONS = ["angry", "scared", "happy", "sad", "surprised","neutral"]
activities = ["angry", "scared", "happy", "sad", "surprised","neutral"] 

angry_emo = []
angry_val = 0
scared_emo = []
scared_val = 0
happy_emo = []
happy_val = 0
sad_emo = []
sad_val = 0
surprised_emo = []
surprised_val = 0
neutral_emo = []
neutral_val = 0

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame and convert it to grayscale
	frame = imutils.resize(frame, width=500)
	original = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# initialize the canvas for the visualization, then clone
	# the frame so we can draw on it
	canvas = np.zeros((200, 200, 3), dtype="uint8")
	frameClone = frame.copy()

	# detect faces in the input frame, then clone the frame so that
	# we can draw on it
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# ensure at least one face was found before continuing
	if len(rects) > 0:
		# determine the largest face area
		rect = sorted(rects, reverse=True,
			key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
		(fX, fY, fW, fH) = rect

		# extract the face ROI from the image, then pre-process
		# it for the network
		roi = gray[fY:fY + fH, fX:fX + fW]
		roi = cv2.resize(roi, (48, 48))
		roi = roi.astype("float") / 255.0
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis=0)

		# make a prediction on the ROI, then lookup the class
		# label
		preds = model.predict(roi)[0]
		label = EMOTIONS[preds.argmax()]

		# loop over the labels + probabilities and draw them
		for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
			# construct the label text
			text = "{}: {:.2f}%".format(emotion, prob * 100)
			emo = emotion
			per = int (prob * 100)
			#print ("%s \t %d" %(emo, per))
			#print ("	#####	")
			if (emo == "neutral"):
				neutral_val = neutral_val + per
			if (emo == "angry"):
				angry_val = angry_val + per
			if (emo == "scared"):
				scared_val = scared_val + per
			if (emo == "happy"):
				happy_val = happy_val + per
			if (emo == "sad"):
				sad_val = sad_val + per
			if (emo == "surprised"):
				surprised_val = surprised_val + per
    
			# draw the label + probability bar on the canvas
			w = int(prob * 100)
			cv2.rectangle(frameClone, (5, (i * 35) + 5),
				(w, (i * 35) + 35), (155, 0, 0), -1)
			cv2.putText(frameClone, text, (10, (i * 35) + 23),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45,
				(255, 255, 255), 2)

		# draw the label on the frame
		cv2.putText(frameClone, label, (fX, fY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
		cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
			(155, 0, 0), 2)
  
		#print (" %d, %d, %d, %d, %d, %d" %(angry_val, scared_val, happy_val, sad_val, surprised_val, neutral_val))

	# show our classifications + probabilities
	#cv2.imshow("Original Frame", original)
	cv2.imshow("Face Emotions", frameClone)
	#cv2.imshow("Probabilities", canvas)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

slices = [angry_val, scared_val, happy_val, sad_val, surprised_val, neutral_val]
  
# color for each label 
colors = ['r', 'y', 'g', 'b', 'c', 'm'] 
  
# plotting the pie chart 
plt.pie(slices, labels = activities, colors=colors,  
        startangle=90, shadow = True, explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1), 
        radius = 1.2, autopct = '%1.1f%%') 
plt.legend()
#plt.show() 
plt.savefig("emotion.jpg", bbox_inches="tight")
print ("done")