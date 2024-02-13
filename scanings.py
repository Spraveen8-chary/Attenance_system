import cv2
import joblib
import os
from datetime import date
from datetime import datetime
import models
import attendance
import numpy as np

DATA = []

name, pin , balance = None,None, None
def qr_data():
    global DATA
    if DATA:
        latest_data = DATA[-1]
        with open("QR_DATA.txt",'w') as file:
            file.write(str(latest_data))
        return latest_data[0], latest_data[1], latest_data[2]
    else:
        return None, None, None

def generate_frames():
    global DATA
    qr_decoder = cv2.QRCodeDetector()

    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            print("Error reading frame from the camera")
            break

        # detect QR code in the frame
        decoded_text, _, _ = qr_decoder.detectAndDecode(frame)

        if decoded_text:
            # extract name, pin, and branch from decoded text
            data = decoded_text

            name_start_index = data.find("Name")
            name_end_index = data.find("Pin")
            name = data[name_start_index:name_end_index].strip().split(":")[1]

            pin_start_index = data.find("Pin")
            pin_end_index = data.find("Branch")
            pin = data[pin_start_index:pin_end_index].strip().split(":")[1]

            branch_start_index = data.find("Branch")
            branch = data[branch_start_index:].strip().split(":")[1]

            # store the extracted data
            DATA.append([name, pin, branch])
            
            # print(DATA)

        else:
            DATA = []

        # encode the frame as a jpeg image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # yield the jpeg image with QR code detection result
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the camera resource
    camera.release()



def face_Recog():
    global name, pin
    # load the face detection model
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # load the face recognition model
    model = joblib.load('static\\face_recognition_model.pkl')

    # initialize lists to store faces and names
    faces = []
    names = []

    # loop through the faces and names in the faces folder
    for user in os.listdir('static\\images\\faces'):
        for imgname in os.listdir(f'static\\images\\faces\\{user}'):
            img = cv2.imread(f'static\\images\\faces\\{user}\\{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face)
            names.append(user)

    
    name, pin, branch = qr_data()
    # start the camera capture
    camera = cv2.VideoCapture(0)
    while True:
        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in the frame
            faces_rects = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            with open('QR_DATA.txt','r') as file:
                content = file.read()
                content = content[1:-1]
                content = eval(content)
                name_pin = f"{content[0]}_{content[1]}"
                # print(name_pin)
                count = 0

            # loop through the faces and draw a rectangle around each face
            for (x, y, w, h) in faces_rects:
                detected_faces = models.extract_faces(frame)
                if detected_faces.any() and len(detected_faces) > 0:
                    (x, y, w, h) = detected_faces[0]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                    face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
                    identified_person = models.identify_face(face.reshape(1, -1))[0]
                    
                    cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,cv2.LINE_AA)

                    current_time = datetime.now().strftime('%H:%M:%S')
                    # evng_attendance(identified_person)
                    count = count + 1
                    if '08:30:00' <= current_time <= '11:33:00' :
                        attendance.mrng_attendance(identified_person)
                    else:
                        attendance.evng_attendance(identified_person)
                    print(f"identified person {identified_person} ==== npstr {name_pin}")
                    if identified_person == np.str_(name_pin):

                        cv2.putText(frame, 'Already Marked...', (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA)
                    else:
                        cv2.putText(frame, 'QR Code Mismatched...', (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA)

            # encode the frame to JPEG format and yield it as a response
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # release the camera and cleanup
    camera.release()
    cv2.destroyAllWindows()