# Psyche-HackJNU

A lot of people suffer from various kind of mental conditions. Be it anxiety, depression or panic attacks. They are more common in teenagers. The pandemic period where everyone were inside their home for more than a year, these mental conditions are becoming more prominent and definetely, need to be addressed.
Our Application, Psyche, helps the people with any kind of mental condition. Giving them a ray of hope.
In our application, these people can register themselves. While registration, all the necessary questions will be asked. They can view the professionals in the field of psychology and can connect with them.

Our Application has two interfaces :
1. Patient Login
2. Doctor Login


Features :
Patient Interface :
1. Doctor's List : The patient will be able to see the professionals and can connect to them once they accept his/her request.
2. Voice Notes : In case, after accepting the request, the patient is not comfortable in sharing his/her condition through offline meeting/video call or gets uncomfortable when someone is around, he can record his voice and then send to doctor.
3. Weekly Analysis : Once the counselling sessions begin, the doctor will send some questions related to patient's condition and the patient has to answer them on camera by himself/herself. The video will be an input to Deep Learning model recording the changes in patient's expression and gives a summary of the expressions at the end of session. This summary will be sent to patient, as well as the consulting doctor for keeping track of the patient.
4. Payment : Once the counselling is complete, the patient can pay the precribed fees to the consulting doctor.
5. Daily Motivation : Based on the input given by the patient while registration, his application home page will be personalized with proper reccomendation of motivational youtube videos, meditation music and relaxation music in the form of audio.

Doctor Interface:
1. Status Updation : The patient can view the incoming requests from all patients and can either accept or decline.
2. Patient Dashboard : There will be a dashboard page wherein all the necessary information about the patient will be visible to the doctor. The doctor can generate meeting link and can send it to patient via mail. From here, he can also send the weekly set of questions to the patient. He can also keep the track of the summary report in the form of graph for his own reference.


Tech Stack :
Frontend : HTML, CSS, JavaScript
Backend : Flask (Python)
Deep Learning : Convolutional Neural Network
Model integration with : OpenCV 
Personalized Reccomendation : Natural Language Processing
Web Scraping : Selenium Bot
Payment : Razorpay Integration

