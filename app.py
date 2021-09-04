from flask.wrappers import Request
from flask import Flask , render_template, request, redirect, url_for, session,Response
import re
import pickle
from flask_mysqldb import MySQL
import MySQLdb.cursors
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
import pyaudio
import wave
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import nltk
from nltk.corpus import stopwords
from selenium import webdriver
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

app=Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'mental health'

# Intialize MySQL
mysql = MySQL(app)
app.secret_key = 'key12'

@app.route('/', methods = ['GET', 'POST'])
def register():
    if request.method == 'POST' :
        # Create variables for easy access
        Name_Patient = request.form['Name_Patient']
        email_Patient = request.form['email_Patient']
        username_Patient = request.form['username_Patient']
        password_Patient = request.form['password_Patient']
        Age_Patient = request.form['Age_Patient']
        Address_Patient = request.form['Address_Patient']
        Phone_number_Patient = request.form['Phone_number_Patient']
        Description_Patient = request.form['Description_Patient']
        Offline = request.form['Offline']
        Video_call = request.form['Video_call']
        Voice_notes = request.form['Voice_notes']
        session['username'] = Name_Patient
        print("a")
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM patient_details WHERE username = %s AND password=%s', [username_Patient, password_Patient])
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
            print("b")
    
        elif not re.match(r'[A-Za-z0-9]+', username_Patient):
            msg = 'Username must contain only characters and numbers!'
            print("c")
        elif not username_Patient or not password_Patient:
            msg = 'Please fill out the form!'
            print("d")
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('INSERT INTO patient_details VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)', [Name_Patient,email_Patient,username_Patient
            ,password_Patient,Age_Patient,Address_Patient,Phone_number_Patient,Description_Patient,Video_call])
            mysql.connection.commit()
            msg = 'Successfully registered! Please Sign-In'
            print("e")
            return redirect(url_for('patient_login'))
            
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    return render_template('register.html')


@app.route('/doc_reg', methods = ['GET', 'POST'])
def doc_reg():
    if request.method == 'POST':
        # Create variables for easy access
        Name_doctor = request.form['Name_doctor']
        email_doctor = request.form['email_doctor']
        username_doctor = request.form['username_doctor']
        password_doctor = request.form['password_doctor']
        Qualification_doctor = request.form['Qualification_doctor']
        jobexp_doctor = request.form['jobexp_doctor']
        Age_doctor = request.form['Age_doctor']
        Clinic_address_doctor = request.form['Clinic_address_doctor']
        Phone_number_doctor = request.form['Phone_number_doctor']
       
        print("a")
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM doctor_details WHERE username = %s AND password=%s', [username_doctor, password_doctor])
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
            print("b")
    
        elif not re.match(r'[A-Za-z0-9]+', username_doctor):
            msg = 'Username must contain only characters and numbers!'
            print("c")
        elif not username_doctor or not password_doctor:
            msg = 'Please fill out the form!'
            print("d")
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('INSERT INTO doctor_details VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)', [Name_doctor,email_doctor,username_doctor,
            password_doctor,Qualification_doctor,jobexp_doctor,Age_doctor,Clinic_address_doctor,Phone_number_doctor])
            mysql.connection.commit()
            msg = 'Successfully registered! Please Sign-In'
            print("e")
            return redirect(url_for('login_doc'))
            
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    return render_template('register.html')


@app.route('/patient_login', methods = ['GET', 'POST'])
def patient_login():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        print("gg")
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM patient_details WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            session['name'] = account['name']
            return redirect(url_for('patient_home'))
    return render_template('login_patient.html')

@app.route('/login_doc', methods = ['GET', 'POST'])
def login_doc():
    if request.method == 'POST':
        print("gg")
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM doctor_details WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            session['name'] = account['name']
            return redirect(url_for('doctor_home'))    
    return render_template('login_doctor.html')


@app.route('/logout_patient', methods = ['GET', 'POST'])
def logout_patient():
    return render_template('login_patient.html')

@app.route('/logout_doctor', methods = ['GET', 'POST'])
def logout_doctor():
    return render_template('login_doctor.html')




@app.route('/doctor_home', methods = ['GET', 'POST'])
def doctor_home():
    Cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    Cursor.execute('SELECT * FROM confirmed_status')
    status_list = Cursor.fetchall()  
    print(status_list)      

    return render_template('doctor_home.html',status_list=status_list)

@app.route('/patient_home', methods = ['GET', 'POST'])
def patient_home():
    Cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    Cursor.execute('SELECT description FROM patient_details')
    status_list = Cursor.fetchall()  
    status_list = list(status_list)

    PATH = "C:/Program Files (x86)/chromedriver.exe"

    driver = webdriver.Chrome(PATH)
    from nltk.tokenize import word_tokenize 
    set(stopwords.words('english'))
    text = str(status_list[-1]['description'])
    stop_words = set(stopwords.words('english')) 

    
    word_tokens = word_tokenize(text) 
        
    filtered_sentence = [] 
    
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 

    filtered="+".join(filtered_sentence)
    print(filtered)
    driver.get("https://www.youtube.com/results?search_query=motivational+video+to+overcome+"+filtered)
    user_data = driver.find_elements_by_xpath('//*[@id="video-title"]')
    links = []
    for i in user_data:
        links.append(i.get_attribute('href'))
        #https://www.youtube.com/embed/tgbNymZ7vqY
    
    new_url_list = list()
    for address in links:
        new_address = address.replace("watch?v=", "embed/")
        new_url_list.append(new_address)

    links = new_url_list
    yt_links = [links[0],links[1],links[2]]
    print(yt_links)
    
    return render_template('patient_home.html',yt_links=yt_links)


@app.route('/doctor_list', methods = ['GET', 'POST'])
def doctor_list():
    Cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    Cursor.execute('SELECT * FROM doctor_details')
    doctor_list = Cursor.fetchall()
    doctor_list = list(doctor_list)
    Doctor_list1 = []
    Doctor_list2 = []
    Doctor_list3 = []
    Doctor_list4 = []
    Doctor_list5 = []
    Doctor_list6 = []


    for i in range(1):
        x = doctor_list[i]
        list_h = []
        for key in x.values():
            list_h.append(key)
        Doctor_list1.append(list_h)

    for i in range(1,2):
        x = doctor_list[i]
        list_h = []
        for key in x.values():
            list_h.append(key)
        Doctor_list2.append(list_h)    

    for i in range(2,3):
        x = doctor_list[i]
        list_h = []
        for key in x.values():
            list_h.append(key)
        Doctor_list3.append(list_h) 


    for i in range(3,4):
        x = doctor_list[i]
        list_h = []
        for key in x.values():
            list_h.append(key)
        Doctor_list4.append(list_h) 

    for i in range(4,5):
        x = doctor_list[i]
        list_h = []
        for key in x.values():
            list_h.append(key)
        Doctor_list5.append(list_h) 

    for i in range(5,6):
        x = doctor_list[i]
        list_h = []
        for key in x.values():
            list_h.append(key)
        Doctor_list6.append(list_h)             

    print(Doctor_list1)    
    print(Doctor_list2)    

    return render_template('doctor_list.html',Doctor_list1=Doctor_list1,Doctor_list2=Doctor_list2,Doctor_list3=Doctor_list3,
    Doctor_list4=Doctor_list4,Doctor_list5=Doctor_list5,Doctor_list6=Doctor_list6)

@app.route('/<name>', methods = ['GET', 'POST'])
def request_status_list(name):
    Status_list = []
    if request.method == "POST":
        Cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        Cursor.execute('SELECT name,day,time FROM doctor_details WHERE name = %s', [name])
        doctor_list = Cursor.fetchall()
        doctor_list = list(doctor_list)
        Doctor_list = []

        x = doctor_list[0]
        list_h = []
        for key in x.values():
            list_h.append(key)
        Doctor_list.append(list_h) 
        name = Doctor_list[0][0]
        day = Doctor_list[0][1]
        time = Doctor_list[0][2]
        status = 'Pending'
        patient_name =session['name']
       

        Cursor.execute('INSERT INTO request_status VALUES(%s,%s,%s,%s,%s)', [name,day,time,status,patient_name])
        mysql.connection.commit()

        Cursor.execute('SELECT * FROM request_status')
        status_list = Cursor.fetchall()
        status_list = list(status_list)
        Status_list = []

        for i in range(len(status_list)):
            x = status_list[i]
            list_h = []
            for key in x.values():
                list_h.append(key)
            Status_list.append(list_h)

     

    return render_template('request_status.html',Status_list=Status_list)

@app.route('/request_status', methods = ['GET', 'POST'])
def request_status():
    Cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    Cursor.execute('SELECT * FROM request_status')
    status_list = Cursor.fetchall()
    status_list = list(status_list)
    Status_list = []

    for i in range(len(status_list)):
            x = status_list[i]
            list_h = []
            for key in x.values():
                list_h.append(key)
            Status_list.append(list_h)

    return render_template('request_status_normal.html',Status_list=Status_list)    

@app.route('/model', methods = ['GET', 'POST'])
def model():
    if request.method == "POST":
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
        cap=cv2.VideoCapture(0)
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
    return  render_template('weekly_analysis.html')


@app.route('/weekly_analysis', methods = ['GET', 'POST'])
def weekly_analysis():
    Cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    if request.method == "POST":
        questions = request.form['questions']
        Cursor.execute('INSERT INTO questions VALUES(%s)', [questions])
        mysql.connection.commit()
        return redirect(url_for('patient_dashboard'))
    Cursor.execute('SELECT * FROM questions')
    questions_list = Cursor.fetchall()
    questions_list = list(questions_list)
    print(questions_list)
    q = questions_list[0]['questions']
    q = q.split('?')
    
    path = 'F:/Codes/Mental Health Assistant/static/graph/'
    image = os.listdir(path)
    if len(image) != 0:
        image = os.listdir(path)[-1]
    else:
        image = 'No results yet!'    
    
    return render_template('weekly_analysis.html',q=q,qwe=image)        

@app.route('/voice_notes', methods = ['GET', 'POST'])
def voice_notes():
    filename = ''
    if request.method == "POST" and request.form['audio_submit'] == 'Submit':
        Cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        Cursor.execute('SELECT doc_name FROM confirmed_status WHERE name = %s', [session['name']])
        status_list = Cursor.fetchone()
        name = status_list['doc_name']
        Cursor.execute('SELECT email FROM doctor_details WHERE name = %s', [name])
        email = Cursor.fetchone()

        email_user = 'parikhdhruv76@gmail.com'
        email_password = 'yddghrhhglieewcx'
        email_send = 'bhamereronit@gmail.com'

        subject = 'Voice Note'

        msg = MIMEMultipart()
        msg['From'] = email_user
        msg['To'] = email_send
        msg['Subject'] = subject

        body = ''
        msg.attach(MIMEText(body,'plain'))


        path = 'F:/Codes/Mental Health Assistant/voice notes/'
        voice = os.listdir(path)
        if len(voice) != 0:
                voice = os.listdir(path)[-1]
                filename= path+voice
                print(filename)
        else:
                image = 'No results yet!'    
        attachment  =open(filename,'rb')

        part = MIMEBase('application','octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',"attachment; filename= "+filename)

        msg.attach(part)
        text = msg.as_string()
        server = smtplib.SMTP('smtp.gmail.com',587)
        server.starttls()
        server.login(email_user,email_password)


        server.sendmail(email_user,email_send,text)
        server.quit()
    
    
    return render_template('voice_notes.html')  

@app.route('/incoming_requests', methods = ['GET', 'POST'])
def incoming_requests():
    Cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    Cursor.execute('SELECT * FROM request_status')
    status_list = Cursor.fetchall()
    status_list = list(status_list)
    Status_list = []

    for i in range(len(status_list)):
            x = status_list[i]
            list_h = []
            for key in x.values():
                list_h.append(key)
            Status_list.append(list_h)
    return render_template('incomingrequest.html',Status_list=Status_list)  


@app.route('/incoming_requests_list', methods = ['GET', 'POST'])
def incoming_requests_list():
    Status_list = []
    if request.method == "POST" :
        name = str(request.form).split(',')[0].replace('ImmutableMultiDict([(','')
        status = str(request.form).split(',')[1].replace(')])','')
        name = name.replace("'",'')
        status = status.replace("'",'')
        print(name, status)
        print(status)

        Cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        Cursor.execute('SELECT * FROM request_status WHERE status = "Pending" and name = %s', [name])
        account = Cursor.fetchone()
        print(account)
        Cursor.execute('UPDATE request_status SET status = %s WHERE name = %s', [status, name])
        mysql.connection.commit()
        print(session['name'])
        if status == " Accept":
            print("hi")
            Cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            Cursor.execute('INSERT INTO confirmed_status VALUES(%s,%s,%s,%s)', [session['name'], account['day'], account['time'],name])
            mysql.connection.commit()
            Cursor.execute('DELETE FROM request_status WHERE status = " Accept"')
            mysql.connection.commit()

        Cursor.execute('SELECT * FROM confirmed_status')
        status_list = Cursor.fetchall()
        status_list = list(status_list)
        Status_list = []

        for i in range(len(status_list)):
                x = status_list[i]
                list_h = []
                for key in x.values():
                    list_h.append(key)
                Status_list.append(list_h)
        return redirect(url_for('confirmed_requests'))        
    return render_template('incomingrequest_normal.html',Status_list=Status_list) 

@app.route('/confirmed_requests', methods = ['GET', 'POST'])
def confirmed_requests():
    Cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    Cursor.execute('SELECT * FROM confirmed_status')
    status_list = Cursor.fetchall()
    return render_template('ConfirmedRequest.html',status_list=status_list)  


@app.route('/patient_dashboard', methods = ['GET', 'POST'])
def patient_dashboard():
    Cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    Cursor.execute('SELECT * FROM patient_details')
    patient_list = Cursor.fetchall()
    patient_list = list(patient_list)
    Patient_list = []

    for i in range(len(patient_list)):
            x = patient_list[i]
            list_h = []
            for key in x.values():
                list_h.append(key)
            Patient_list.append(list_h)

    path = 'F:/Codes/Mental Health Assistant/static/graph/'
    image = os.listdir(path)
    if len(image) != 0:
        image = os.listdir(path)[-1]
    else:
        image = 'No results yet!' 
    
    return render_template('patient_dashboard.html',Patient_list=Patient_list,image=image)   
import gdown

@app.route('/video_analysis', methods = ['GET', 'POST'])
def video_analysis():
    if request.method == "POST":
        link = request.form['link']
        new = link.split('/')
        link = 'https://drive.google.com/uc?id='+str(new[5])
        output = 'video.mp4'
        gdown.download(link, output, quiet=False)


    return redirect(url_for('patient_dashboard'))

@app.route('/video', methods = ['GET', 'POST'])
def video():
    PATH = "C:\Program Files (x86)\chromedriver.exe"
    driver = webdriver.Chrome(PATH)
    driver.get("https://talky.io/")
    button = driver.find_element_by_class_name("create-room-form-button")
    button.click()
    link=driver.current_url
    print(link)
    return 'The link for your session with the patient is: '+ link + '. You can copy this link and send it to patient whenever your call is scheduled'

import razorpay
client = razorpay.Client(auth=("rzp_test_edCZyVYZGMDSwH","x6Okvnq47TjI8OtQDS0dktPC"))
@app.route('/payment', methods = ['GET', 'POST'])
def payment():
    name_of_event = 'example'
    amount = 500 * 100
    payment = client.order.create({'amount' : amount, 'currency' : 'INR', 'payment_capture' : '1'})
    event_details = [name_of_event]
    if request.form:
        
        amount = int(request.form['amount']) * 100
        payment = client.order.create({'amount' : amount, 'currency' : 'INR', 'payment_capture' : '1'})
        event_details = [name_of_event]
        return render_template('pay.html',event_details=event_details,payment=payment)
    return render_template('pay.html',event_details=event_details,payment=payment)

import keyboard
@app.route('/record_audio', methods = ['GET', 'POST'])
def record_audio():
    audio = pyaudio.PyAudio()
    from datetime import datetime
    now = datetime.now()
    path = 'F:/Codes/Mental Health Assistant/voice notes/'
    dt_string = now.strftime("%d%m%Y_%H%M%S")

    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    frames = []
    filename = ''
    if request.method == "POST" and request.form['audio_submit'] == 'Submit':
        email_user = 'parikhdhruv76@gmail.com'
        email_password = 'yddghrhhglieewcx'
        email_send = 'bhamereronit@gmail.com'

        subject = 'Voice Note'

        msg = MIMEMultipart()
        msg['From'] = email_user
        msg['To'] = email_send
        msg['Subject'] = subject

        body = ''
        msg.attach(MIMEText(body,'plain'))


        path = 'F:/Codes/Mental Health Assistant/voice notes/'
        voice = os.listdir(path)
        if len(voice) != 0:
                voice = os.listdir(path)[-1]
                filename= path+voice
                print(filename)
        else:
                image = 'No results yet!'    
        attachment  =open(filename,'rb')

        part = MIMEBase('application','octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',"attachment; filename= "+filename)

        msg.attach(part)
        text = msg.as_string()
        server = smtplib.SMTP('smtp.gmail.com',587)
        server.starttls()
        server.login(email_user,email_password)


        server.sendmail(email_user,email_send,text)
        server.quit()
    while True :
        data = stream.read(1024)
        frames.append(data)

        if keyboard.is_pressed("q"):
                print('hi')

                stream.stop_stream()
                stream.close()
                audio.terminate()
                filename = path+str(dt_string)+".wav"
                sound_file = wave.open(filename, "wb")
                sound_file.setnchannels(1)
                sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                sound_file.setframerate(44100)
                sound_file.writeframes(b''.join(frames))
                sound_file.close()
                return render_template('voice_notes.html')


if __name__=="__main__":
    app.run(debug=True)        