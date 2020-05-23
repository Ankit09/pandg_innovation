# pandg_innovation
 Detecting consumer product feedback using customers' reaction by using face expression as well doing customer segmentation based on age & gender


Detecting consumer product feedback using customers' reaction by using face expression as well doing customer segmentation based on age & gender

The whole project is devided into 3 parts.

Face detection & Facial Landmark detection. (It will be used for next 2 steps)
Customers' face expression detection and stored the time for the further analysis and graph representation.
Customers' age and gender detection for customers' segmentation analysis for the product.
Let's understand the project structure. 1-facial-points-realtime 2-emotion-realtime 3-AgeGender-realtime master.py

we need to run the master.py file to run the whole project:

python master.py

To understand each files seprately, we need to run this one by one like this

Face detection & Facial Landmark detection. (It will be used for next 2 steps) cd 1-facial-points-realtime/ and run the file python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat

Customers' face expression detection and stored the time for the further analysis and graph representation. cd 2-emotion-realtime and run the file python emotion_detector.py --cascade haarcascade_frontalface_default.xml --model checkpoints/epoch_75.hdf5

Customers' age and gender detection for customers' segmentation analysis for the product. cd 3-AgeGender-realtime python AgeGender.py

Thank You.
