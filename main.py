import qrcode
import os 
from flask import Flask,render_template,request,url_for,redirect,Response,stream_with_context
import scanings
import cv2
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import datetime
from datetime import date
from datetime import datetime

app = Flask(__name__)
# app.use_static_folder('static')
camera=cv2.VideoCapture(0)
qrs = []

# routing part
database =[]



name, pin, branch = None, None, None




@app.route('/')
def home():
    return render_template('home.html')

@app.route('/qrcode', methods=['POST','GET'])
def register():
    if request.method == 'POST':
        name = request.form['Name'].upper()
        pin = request.form['pin'].upper()
        branch = request.form['branch'].upper()
        

        data = f'''
                Name    :{name}
                Pin     :{pin}
                Branch  :{branch}'''
        qr = qrcode.make(data)
        qr.save(f"static\\images\\QR_CODES\\{name}_{pin}.png")

        
    return render_template("qr.html")
    



    

@app.route('/database', methods = (['POST','GET']))
def db():
    name= request.form['name'].upper()
    pin = request.form['pin'].upper()
    branch = request.form['branch'].upper()

    fetch = f"{name}_{pin}.png"
    total_qrs = os.listdir("static\\images\\students")

    if fetch not in total_qrs:
        fetch = "no_user.png"

    return render_template('data.html', name=name, pin=pin, branch=branch, fetch=fetch)


@app.route('/validate')
def valid():
    return render_template('validation.html')



def generate_frames():
    global name, pin, branch
    
    # create QR code detector object
    qr_decoder = cv2.QRCodeDetector()

    while True:
        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
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

                # print the extracted data
                print("Name:", name)
                print("Pin:", pin)
                print("Branch:", branch)
                break


            else:
                name, pin, branch = None, None, None

            # encode the frame as a jpeg image
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # yield the jpeg image with QR code detection result
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            index()


@app.route('/qr_Scan')
def index():
    name, pin, branch = scanings.qr_data()
    fetch = "no_user.png"
    print("Before scan : ", name, pin, branch, fetch)

    if name is not None and pin is not None and branch is not None:
        fetch = f"{name}_{pin}.png"
        total_qrs = os.listdir("static/images/QR_CODES")

        if fetch not in total_qrs:
            fetch = "no_user.png"

        print("After scanning : ", name, pin, branch, fetch)

        return render_template('streaming.html', name=name, pin=pin, branch=branch, fetch=fetch)
    else:
        return render_template('streaming.html', name=name, pin=pin, branch=branch, fetch=fetch)


@app.route('/video')
def video():
    return Response(stream_with_context(scanings.generate_frames()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# define the face detector model
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#### extract the face from an image
def add_faces():
    global name, pin
    i= 0
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            face=face_detector.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
            for x,y,w,h in face:
                # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
                cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,cv2.LINE_AA)
                if name is not None and pin is not None:
                    user_dir = os.path.join("static\\images\\faces\\", name + "_" + pin)
                    if not os.path.exists(user_dir):
                        os.makedirs(user_dir)
                    img_path = os.path.join(user_dir, f"{name}_{i}.jpg")
                    cv2.imwrite(img_path, frame[y:y+h,x:x+w])
                
                i += 1
            if i >= 50:
                break

            train_model()
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/face_validation', methods=['POST', 'GET'])
def adding_face():
    global name,pin
    if request.method == 'POST':
        # get the form data
        name   = request.form['Name'].upper()
        pin    = request.form['pin'].upper()
        branch = request.form['branch'].upper()

        # create the directory for the user
        user_dir = os.path.join("static\\images\\faces\\", name + "_" + pin)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        # start the camera and capture 50 images of the user's face
        return render_template('faces.html')

    # if the request method is not POST, show the add_face.html form
    return render_template('add_face.html')

@app.route('/faces')
def faces():
    return Response(add_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/live_video')
def live_video():
    return render_template('faces.html')
# ===================================================================================================
# ATTENDANCE PART 
# ============================================================================


curent_time = date.today().strftime("%d_%m_%y")
def excels():
#  evening attendance 
    if f'Attendance\\evng_attendance\\Attendance-{curent_time}.csv' not in os.listdir('Attendance\\evng_attendance'):
        with open(f'Attendance\\evng_attendance\\Attendance-{curent_time}.csv', 'w') as f:
            f.write('Name,Pin,Branch,Final_time')

    # morning attendace
    if f'Attendance\\mrng_attendance\\Attendance-{curent_time}.csv' not in os.listdir('Attendance\\mrng_attendance'):
        with open(f'Attendance\\mrng_attendance\\Attendance-{curent_time}.csv', 'w') as f:
            f.write('Name,Pin,Branch,Initial_time')

    # final attendance for calculations

    if f'Attendance\\final_attendance\\Attendance-{curent_time}.csv' not in os.listdir('Attendance\\final_attendance'):
        with open(f'Attendance\\final_attendance\\Attendance-{curent_time}.csv', 'w') as f:
            f.write('Name,Pin,Branch,Initial_time,Final_time,Time_difference,Status')

excels()
file1 = pd.read_csv(f'Attendance\\mrng_attendance\\Attendance-{curent_time}.csv')
file2 = pd.read_csv(f'Attendance\\evng_attendance\\Attendance-{curent_time}.csv')

merged_data = pd.merge(file1,file2,on=['Name','Pin','Branch'],how = 'outer')

merged_data['Time_difference']= pd.to_datetime(merged_data['Final_time'])-pd.to_datetime(merged_data['Initial_time'])

merged_data['Time_difference'] = merged_data['Time_difference']/pd.Timedelta(hours=1)

merged_data['Status'] = ''

merged_data.loc[merged_data['Time_difference']<6,'Status'] = 'Half-Day'

merged_data.loc[merged_data['Time_difference']>6,'Status'] = 'Full-Day'

merged_data.loc[merged_data['Time_difference']<3 ,'Status'] = 'ERROR'

merged_data.to_csv(f'Attendance\\final_attendance\\Attendance-{curent_time}.csv',index=False)

#### Add Attendance of a specific user
def mrng_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")


    df = pd.read_csv(f'Attendance\\mrng_attendance\\Attendance-{curent_time}.csv')
    if str(userid) not in list(df['Pin']):
        with open(f'Attendance\\mrng_attendance\\Attendance-{curent_time}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{branch},{current_time}')


def evng_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    att = {
        'Name' : [username],
        'Pin': [userid],
        'Branch':[branch],
        'Final_time': [current_time]
    }
    data = pd.DataFrame(att)

    df = pd.read_csv(f'Attendance\\evng_attendance\\Attendance-{curent_time}.csv')
    if str(userid) not in list(df['Pin']):
        with open(f'Attendance\\evng_attendance\\Attendance-{curent_time}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{branch},{current_time}')


def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points
#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static\\face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static\\images\\faces')
    for user in userlist:
        for imgname in os.listdir(f'static\\images\\faces\\{user}'):
            img = cv2.imread(f'static\\images\\faces\\{user}\\{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static\\face_recognition_model.pkl')

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

            # loop through the faces and draw a rectangle around each face
            for (x, y, w, h) in faces_rects:
                detected_faces = extract_faces(frame)
                if detected_faces and len(detected_faces) > 0:
                    (x, y, w, h) = detected_faces[0]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                    face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
                    identified_person = identify_face(face.reshape(1, -1))[0]
                    current_time = datetime.now().strftime('%H:%M:%S')
                    # evng_attendance(identified_person)
                    if '09:00:00' <= current_time <= '15:56:00' :
                        mrng_attendance(identified_person)
                    else:
                        evng_attendance(identified_person)
                    cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,cv2.LINE_AA)
                    
            # encode the frame to JPEG format and yield it as a response
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # release the camera and cleanup
    camera.release()
    cv2.destroyAllWindows()


@app.route('/face_attendance')
def face_attendance():
    return render_template('face_att.html')

@app.route('/face_att')
def face_att():
    return Response(face_Recog(), mimetype='multipart/x-mixed-replace; boundary=frame')

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)



