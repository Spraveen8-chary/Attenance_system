from flask import Flask,render_template,request,url_for,redirect,Response,stream_with_context
import os
import qrcode
import pandas as pd
import datetime
from datetime import date
from datetime import datetime
import cv2
import models
import scanings
import attendance
import connect_database


attendance.excels()

app = Flask(__name__)

name, pin, branch = None, None, None


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/qrcode', methods = ['GET','POST'])
def qr_code_generation():
    if request.method == 'POST':
        name= request.form['name'].upper()
        pin = request.form['pin'].upper()
        branch = request.form['branch'].upper()
        print(name, pin , branch)

        data = f'''
                Name    :{name}
                Pin     :{pin}
                Branch  :{branch}'''
        qr = qrcode.make(data)
        qr.save(f"static\\images\\QR_CODES\\{name}_{pin}.png")

    return render_template('qr.html')



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


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

save_path = "static/images/faces/"

@app.route('/add_face', methods=['GET', 'POST'])
def add_face():
    if request.method == 'POST':

        name = request.form['name'].upper()
        pin = request.form['pin'].upper()
        branch = request.form['branch'].upper()

        user_directory = os.path.join(save_path, f"{name}_{pin}")
        os.makedirs(user_directory, exist_ok=True)

        cap = cv2.VideoCapture(0)

        img_count = 0

        while True:

            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.putText(frame, f"Image Count: {img_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Capture Face', frame)

            img_path = os.path.join(user_directory, f"{name}_{pin}_{img_count}.jpg")
            cv2.imwrite(img_path, frame)

            img_count += 1

            if img_count == 75 :
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        models.train_model()
        connect_database.insert_images(user_name=f'{name}_{pin}')
        
        cap.release()
        cv2.destroyAllWindows()

        return redirect(url_for('home'))

    return render_template('add_face.html')



@app.route('/validate')
def valid():
    return render_template('validation.html')



@app.route('/qr_Scan')
def index():
    name, pin, branch = scanings.qr_data()
    fetch = "no_user.png"
    print("Before scan : ",name,pin,branch,fetch)
    
    if name is not None and pin is not None and branch is not None:
        fetch = f"{name}_{pin}.png"
        total_qrs = os.listdir("static/images/students")

        if fetch not in total_qrs:
            fetch = "no_user.png"
        
        print("After scanning : ",name,pin,branch,fetch)

        return render_template('streaming.html', name=name, pin=pin, branch=branch, fetch=fetch)
    else:
        return render_template('streaming.html', name=name, pin=pin, branch=branch, fetch=fetch) 


@app.route('/video')
def video():
    return Response(stream_with_context(scanings.generate_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/face_attendance')
def face_attendance():
    return render_template('face_att.html')



@app.route('/face_att')
def face_att():
    return Response(stream_with_context(scanings.face_Recog()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/live_video')
def live_video():
    return render_template('faces.html')






@app.route("/signup", methods=['GET',"POST"])
def signup():
    user_email ="email@gmail.com"
    user_password = "password"
    email = request.form.get('email')
    password = request.form.get('password')
    if email == user_email and password == user_password:
        return morning()
    return render_template('signup_page.html')

@app.route("/signin", methods=['GET',"POST"])
def signin():
    return render_template('signin_page.html')


@app.route('/morning')
def morning():
    today_date_str = date.today().strftime("%d_%m_%y")
    excel_file = f'Attendance\\mrng_attendance\\Attendance-{today_date_str}.csv'
    df = pd.read_csv(excel_file)
    data = df.to_html(classes='table table-striped', index=False)
    department = "computer science engineering"
    current_date = datetime.now().strftime("%d/%m/%Y")
    return render_template('morning.html', data=data, date=current_date, active="active", department=department)


@app.route("/evening", methods=['GET',"POST"])
def evening():
    today_date_str = date.today().strftime("%d_%m_%y")
    excel_file = f'Attendance\\evng_attendance\\Attendance-{today_date_str}.csv'
    df = pd.read_csv(excel_file)
    data = df.to_html(classes='table table-striped', index=False)
    department = "computer science engineering".upper()
    current_date = datetime.now().strftime("%d/%m/%Y")
    return render_template('evening.html', data=data, date=current_date, active="active", department=department)


@app.route("/total", methods=['GET',"POST"])
def total():
    today_date_str = date.today().strftime("%d_%m_%y")
    excel_file = f'Attendance\\final_attendance\\Attendance-{today_date_str}.csv'
    df = pd.read_csv(excel_file)
    data = df.to_html(classes='table table-striped', index=False)
    print(data)
    department = "computer science engineering".upper()
    current_date = datetime.now().strftime("%d/%m/%Y")
    return render_template('total_attandence.html', data=data, date=current_date, active="active", department=department)


@app.route("/statistics", methods=['GET',"POST"])
def statistics():
    department="computer science engineering".upper()
    date = datetime.now().strftime("%d/%m/%Y")
    return render_template('statistics.html',active="active",date=date,department=department)




if __name__ == '__main__':
    app.run(debug=True)