import cv2
import numpy as np
from PIL import ImageGrab 
import os
import time
emotion_path = "emotion.jpg"
gender_path = "male_female.jpg"
m_path = "male_age.jpg"
f_path = "female_age.jpg"

#facial landmarks
os.system("python 1-facial-points-realtime/facial_landmarks.py --shape-predictor 1-facial-points-realtime/shape_predictor_68_face_landmarks.dat &")
time.sleep(1)
os.system("python 2-emotion-realtime/emotion_detector.py --cascade 2-emotion-realtime/haarcascade_frontalface_default.xml --model 2-emotion-realtime/checkpoints/epoch_75.hdf5 &")
time.sleep(1)
os.system("python 3-AgeGender-realtime/AgeGender.py")
emotion_image = cv2.imread(emotion_path)
gender_image = cv2.imread(gender_path)
m_image = cv2.imread(m_path)
f_image = cv2.imread(f_path)
cv2.imshow("Female Visitors statistics", f_image)
cv2.imshow("Male Visitors statistics", m_image)
cv2.imshow("Male & Female Visitors Statistics", gender_image)
cv2.imshow("Visitors' Reaction to Product", emotion_image)
cv2.waitKey(0)