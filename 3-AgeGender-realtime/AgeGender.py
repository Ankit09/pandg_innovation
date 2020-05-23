# Import required modules
import cv2 as cv
import math
import time
import argparse
import matplotlib.pyplot as plt_gender
import matplotlib.pyplot as m_age
import matplotlib.pyplot as f_age
import imutils
#to get the face
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

#arguments
parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

args = parser.parse_args()

faceProto = "3-AgeGender-realtime/opencv_face_detector.pbtxt"
faceModel = "3-AgeGender-realtime/opencv_face_detector_uint8.pb"

ageProto = "3-AgeGender-realtime/age_deploy.prototxt"
ageModel = "3-AgeGender-realtime/age_net.caffemodel"

genderProto = "3-AgeGender-realtime/gender_deploy.prototxt"
genderModel = "3-AgeGender-realtime/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList          = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
m_activities_age = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
f_activities_age = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

m_one_age   = 0
m_two_age   = 0
m_three_age = 0
m_four_age  = 0
m_five_age  = 0
m_six_age   = 0
m_seven_age = 0
m_eight_age = 0

f_one_age   = 0
f_two_age   = 0
f_three_age = 0
f_four_age  = 0
f_five_age  = 0
f_six_age   = 0
f_seven_age = 0
f_eight_age = 0

genderList        = ['Male', 'Female']
activities_gender = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

# Open a video file or an image file or a camera stream
cap = cv.VideoCapture(args.input if args.input else 0)
padding = 20
total_time_male = 0
total_time_female = 0 
while cv.waitKey(1) < 0:
    # Read frame
    t = time.time()
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue

    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        # print("Gender Output : {}".format(genderPreds))
        #print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        #print("Age Output : {}".format(agePreds))
        #print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

        label = "{},{}".format(gender, age)
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        frame_resized = imutils.resize(frameFace, width=500)
        cv.imshow("Spectators' Age Gender Display", frame_resized)
        # cv.imwrite("age-gender-out-{}".format(args.input),frameFace)
    total_time = ("time : {:.3f}".format(time.time() - t))
    total_time = ("{:.3f}".format(time.time() - t))
    float_time = float(total_time)
    
    if (gender == "Male"):
        print ("male detected")
        total_time_male = total_time_male + float_time

        if (age == "(0-2)"):
            m_one_age = m_one_age + float_time
        if (age == "(4-6)"):
            m_two_age = m_two_age + float_time
        if (age == "(8-12)"):
            m_three_age = m_three_age + float_time
        if (age == "(15-20)"):
            m_four_age = m_four_age + float_time
        if (age == "(25-32)"):
            m_five_age = m_five_age + float_time
        if (age == "(38-43)"):
            m_six_age = m_six_age + float_time
        if (age == "(48-53"):
            m_seven_age = m_seven_age + float_time
        if (age == "(60-100)"):
            m_eight_age = m_eight_age + float_time
        
    if (gender == "Female"):
        print ("female detected")
        total_time_female = total_time_female + float_time

        if (age == "(0-2)"):
            f_one_age = f_one_age + float_time
        if (age == "(4-6)"):
            f_two_age = f_two_age + float_time
        if (age == "(8-12)"):
            f_three_age = f_three_age + float_time
        if (age == "(15-20)"):
            f_four_age = f_four_age + float_time
        if (age == "(25-32)"):
            f_five_age = f_five_age + float_time
        if (age == "(38-43)"):
            f_six_age = f_six_age + float_time
        if (age == "(48-53"):
            f_seven_age = f_seven_age + float_time
        if (age == "(60-100)"):
            f_eight_age = f_eight_age + float_time

#total_time_male = int(total_time_male)
#total_time_male = int(total_time_female)
print ("Total male: %d" %(total_time_male))
print ("Total female: %d" %(total_time_female))

m_one_age   = int(m_one_age)
m_two_age   = int(m_two_age)
m_three_age = int(m_three_age)
m_four_age  = int(m_four_age)
m_five_age  = int(m_five_age)
m_six_age   = int(m_six_age)
m_seven_age = int(m_seven_age)
m_eight_age = int(m_eight_age)

print ("%d \t  %d \t %d \t %d \t %d \t %d \t %d \t %d \t" %(m_one_age, m_two_age, m_three_age, m_four_age ,m_five_age, m_six_age, m_seven_age, m_eight_age))
total_male_count  = m_one_age + m_two_age + m_three_age + m_four_age + m_five_age + m_six_age + m_seven_age + m_eight_age
print ("Total Male %d" %(total_male_count))

f_one_age = int(f_one_age)
f_two_age = int(f_two_age)
f_three_age = int(f_three_age)
f_four_age = int(f_four_age)
f_five_age = int(f_five_age)
f_six_age = int(f_six_age)
f_seven_age = int(f_seven_age)
f_eight_age = int(f_eight_age)

print ("%d \t  %d \t %d \t %d \t %d \t %d \t %d \t %d \t" %(f_one_age, f_two_age, f_three_age, f_four_age ,f_five_age, f_six_age, f_seven_age, f_eight_age))
total_female_count  = f_one_age + f_two_age + f_three_age + f_four_age + f_five_age + f_six_age + f_seven_age + f_eight_age
print ("Total f_female %d" %(total_female_count))


f_slices_age = [f_one_age, f_two_age, f_three_age, f_four_age, f_five_age, f_six_age, f_seven_age, f_eight_age]
m_slices_age = [m_one_age, m_two_age, m_three_age, m_four_age, m_five_age, m_six_age, m_seven_age, m_eight_age]

slices_gender = [total_male_count, total_female_count]
# color for each label 
colors_gender = ['r', 'y'] 
colors_male = ['r', 'y', 'g', 'b', 'r', 'c', 'm', 'w'] 
colors_female = ['r', 'y', 'g', 'b', 'r', 'c', 'm', 'w'] 
 
''' 
fig = plt.figure()
ax1 = fig.add_axes([0, .5, .5, .5], aspect=1)
ax1.pie(slices_gender, labels = activities_gender, colors=colors, startangle=90, explode = (0.1, 0.1), radius = 1.2, autopct = '%1.1f%%') 
ax2 = fig.add_axes([.5, .0, .5, .5], aspect=1)
ax2.pie(m_slices_age, labels = m_activities_age, colors=colors, startangle=90, explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), radius = 1.2, autopct = '%1.1f%%') 
ax3 = fig.add_axes([.0, .0, .5, .5], aspect=1)
ax3.pie(f_slices_age, labels = f_activities_age, colors=colors, startangle=90, explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), radius = 1.2, autopct = '%1.1f%%') 

ax1.set_title('Male & Female Spectators record')
ax2.set_title('Male spectators record')
ax3.set_title('Female spectators record')
'''
plt_gender.pie(slices_gender, labels = activities_gender, colors=colors_gender, startangle=90, explode = (0.0, 0.0), radius = 1.2, autopct = '%1.1f%%') 
plt_gender.legend()
#plt_gender.title("Male & Female visitor statistics")
plt_gender.savefig("male_female.jpg", bbox_inches="tight")

m_age.pie(m_slices_age, labels = m_activities_age, colors=colors_male, startangle=90, explode = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), radius = 1.2, autopct = '%1.1f%%') 
m_age.legend()
#m_age.title("Male Visitor statistics")
m_age.savefig("male_age.jpg", bbox_inches="tight")

f_age.pie(f_slices_age, labels = f_activities_age, colors=colors_female, startangle=90, explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), radius = 1.2, autopct = '%1.1f%%') 
f_age.legend()
#f_age.title("Female Visitor statistics")
f_age.savefig("female_age.jpg", bbox_inches="tight")
