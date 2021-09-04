import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
now = datetime.now()
model = model_from_json(open("fer.json", "r").read())
model.load_weights('CNN.50-0.67.hdf5')


face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

angry=0
disgust=0
fear=0
happy=0
sad=0 
surprise=0
neutral=0
count=0
print('hey')
cap=cv2.VideoCapture('video.mp4')
print('hi')

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
 

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])
        predictions = predictions[0]
        #print(predictions)
        angry+=predictions[0]
        disgust+=predictions[1]
        fear+=predictions[2]
        happy+=predictions[3]
        sad+=predictions[4]
        surprise+=predictions[5]
        neutral+=predictions[6]
        count+=1
        #print(max_index)
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)

    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows

angry=round(angry/count,2)
disgust=round(disgust/count,2)
fear=round(fear/count,2)
happy=round(happy/count,2)
sad=round(sad/count,2)
surprise=round(surprise/count,2)
neutral=round(neutral/count,2)
print(count)
print(angry+disgust+fear+happy+sad+surprise+neutral)

avgs = [angry, disgust, fear, happy, sad, surprise, neutral]
values = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

print('hello')

plt.bar(values, avgs, color ='maroon',
        width = 0.4)

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
  
plt.xlabel("Emotions")
plt.ylabel("Probability")

addlabels(values, avgs)
dt_string = now.strftime("%d%m%Y_%H%M%S")
filename = 'F:/Codes/Mental Health Assistant/static/graph/'+str(dt_string)+'.png'
plt.savefig(filename)  
plt.show()
