import os
import cv2
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static\\images\\faces')

    for user in userlist:
        for imgname in os.listdir(f'static\\images\\faces\\{user}'):
            try:
                img = cv2.imread(f'static\\images\\faces\\{user}\\{imgname}', cv2.IMREAD_GRAYSCALE)
                resized_face = cv2.resize(img, (50, 50))
                normalized_face = resized_face / 255.0  
                faces.append(normalized_face.ravel())
                labels.append(user)
            except Exception as e:
                print(f"Error processing image {imgname}: {str(e)}")

    faces = np.array(faces)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2,random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    joblib.dump(knn, 'static\\face_recognition_model.pkl')



def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points



#### Identify face using ML model

def identify_face(facearray):

    if len(facearray.shape) == 2:
        gray_face = facearray
    else:
        gray_face = cv2.cvtColor(facearray, cv2.COLOR_BGR2GRAY)

    resized_face = cv2.resize(gray_face, (50, 50))

    flattened_face = resized_face.ravel()

    reshaped_face = flattened_face.reshape(1, -1)

    model = joblib.load('static\\face_recognition_model.pkl')

    return model.predict(reshaped_face)


train_model()

