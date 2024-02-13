# import cv2
# import os
# import sys
# import cv2
# from tensorflow.keras.preprocessing.image import img_to_array
# import os
# import numpy as np
# from tensorflow.keras.models import model_from_json
# import joblib
# from datetime import datetime


# root_dir = os.getcwd()
# # Load Face Detection Model
# face_cascade = cv2.CascadeClassifier("anti_spoof/models/haarcascade_frontalface_default.xml")
# # Load Anti-Spoofing Model graph
# json_file = open('anti_spoof/antispoofing_models/antispoofing_model.json','r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load antispoofing model weights 
# model.load_weights('anti_spoof/antispoofing_models/antispoofing_model.h5')
# print("Model loaded from disk")


# video = cv2.VideoCapture(0)
# while True:
#     try:
#         ret,frame = video.read()
#         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray,1.3,5)
#         for (x,y,w,h) in faces:  
#             face = frame[y-5:y+h+5,x-5:x+w+5]
#             resized_face = cv2.resize(face,(160,160))
#             resized_face = resized_face.astype("float") / 255.0
#             # resized_face = img_to_array(resized_face)
#             resized_face = np.expand_dims(resized_face, axis=0)
#             # pass the face ROI through the trained liveness detector
#             # model to determine if the face is "real" or "fake"
#             preds = model.predict(resized_face)[0]
#             print(preds)
#             if preds> 0.5:
#                 label = 'spoof'
#                 cv2.putText(frame, label, (x,y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
#                 cv2.rectangle(frame, (x, y), (x+w,y+h),
#                     (0, 0, 255), 2)
#             else:
#                 label = 'real'
#                 cv2.putText(frame, label, (x,y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
#                 print("REAL.......")
#                 cv2.rectangle(frame, (x, y), (x+w,y+h),
#                 (0, 255, 0), 2)
#         cv2.imshow('frame', frame)
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
#     except Exception as e:
#         pass
# video.release()        
# cv2.destroyAllWindows()
# # =================================================================
import csv
from datetime import datetime, timedelta
from datetime import date
import pandas as pd
import os

import datetime
import datetime

initial_time = '05:12:53'
# print(initial_time)

# Parse the time string
hours, minutes, seconds = map(int, initial_time.split(':'))

# Get today's date
today_date = datetime.date.today()

# Combine today's date with the parsed time
combined_datetime = datetime.datetime.combine(today_date, datetime.time(hours, minutes, seconds))
# print(combined_datetime)



import pandas as pd
today_date = date.today().strftime("%d_%m_%y")
csv_file_path = f'Attendance\\final_attendance\\Attendance-{today_date}.csv'
data = pd.read_csv(csv_file_path)

# print(data)
Name = ''
Pin = ''
Branch = ''
Initial_time = ''
Final_time = ''
Time_difference = 0
Status = ''



import csv
with open(csv_file_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        name, pin, branch, initial_time, final_time, time_difference, status = row
        print(name, pin, branch, initial_time, final_time, time_difference, status)
