from datetime import date
from datetime import datetime
import os
import pandas as pd
import scanings
import cv2
import connect_database

today_date = date.today().strftime("%d_%m_%y")
def excels():
#  evening attendance 
    if f'Attendance\\evng_attendance\\Attendance-{today_date}.csv' not in os.listdir('Attendance\\evng_attendance'):
        with open(f'Attendance\\evng_attendance\\Attendance-{today_date}.csv', 'w') as f:
            f.write('Name,Pin,Branch,Final_time')

    # morning attendace
    if f'Attendance\\mrng_attendance\\Attendance-{today_date}.csv' not in os.listdir('Attendance\\mrng_attendance'):
        with open(f'Attendance\\mrng_attendance\\Attendance-{today_date}.csv', 'w') as f:
            f.write('Name,Pin,Branch,Initial_time')

    # final attendance for calculations

    if f'Attendance\\final_attendance\\Attendance-{today_date}.csv' not in os.listdir('Attendance\\final_attendance'):
        with open(f'Attendance\\final_attendance\\Attendance-{today_date}.csv', 'w') as f:
            f.write('Name,Pin,Branch,Initial_time,Final_time,Time_difference,Status')

def read_qr_code(filename):
    
    try:
        img = cv2.imread(filename)
        detect = cv2.QRCodeDetector()
        value, points, straight_qrcode = detect.detectAndDecode(img)
        
        branch_start_index = value.find("Branch")
        branch = value[branch_start_index:].strip().split(":")[1]
        return branch
    except:
        return
    

#### Add Attendance of a specific user
def mrng_attendance(name):

    branch = read_qr_code(f'static\\images\\QR_CODES\\{name}.png')
    print(branch)
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    with open('QR_DATA.txt','r') as  file:
        content = file.read()
        # print(content)
        content = content[1:-1]
        content = eval(content)

        n = content[0]
        p = content[1]
        b = content[2]

        print(f"File Content: {n,p,b}")
        
        if str(username) == n.strip() and str(userid) == p.strip() and str(branch) == b.strip():

            print("DATA MARTCHED..........")

            df = pd.read_csv(f'Attendance\\mrng_attendance\\Attendance-{today_date}.csv')
            if str(userid) not in list(df['Pin']):
                with open(f'Attendance\\mrng_attendance\\Attendance-{today_date}.csv', 'a') as f:
                    f.write(f'\n{username},{userid},{branch},{current_time}')
            connect_database.mrng_csv(username = username,userid= userid,branch = branch,current_time= current_time)
            print("ATTENDANCE MARKED.....")
        else:
            print("Mismatched.....")
           



def evng_attendance(name):
    branch = read_qr_code(f'static\\images\\QR_CODES\\{name}.png')
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    # print(username,userid,branch)
    with open('QR_DATA.txt','r') as  file:
        content = file.read()
        # print(content)
        content = content[1:-1]
        content = eval(content)
        print(content)
        n = content[0]
        p = content[1]
        b = content[2]

        print(f"File Content: {n,p,b}")
        print(username,userid,branch)

        if str(username) == n.strip() and str(userid) == p.strip() and str(branch) == b.strip():

            print("DATA MARTCHED..........")
            
            df = pd.read_csv(f'Attendance\\evng_attendance\\Attendance-{today_date}.csv')
            if str(userid) not in list(df['Pin']):
                with open(f'Attendance\\evng_attendance\\Attendance-{today_date}.csv', 'a') as f:
                    f.write(f'\n{username},{userid},{branch},{current_time}')
            
            final_attendance()
            connect_database.evng_csv(username = username,userid= userid,branch = branch,current_time= current_time)
            # connect_database.final_csv()



        else:
            print("MIS MATCH....")



def final_attendance():
    today_date = date.today().strftime("%d_%m_%y")

    file1 = pd.read_csv(f'Attendance\\mrng_attendance\\Attendance-{today_date}.csv')

    file2 = pd.read_csv(f'Attendance\\evng_attendance\\Attendance-{today_date}.csv')

    merged_data = pd.merge(file1,file2,on=['Name','Pin','Branch'],how = 'outer')

    merged_data['Time_difference']= pd.to_datetime(merged_data['Final_time'])-pd.to_datetime(merged_data['Initial_time'])

    merged_data['Time_difference'] = merged_data['Time_difference'].dt.total_seconds() / 60

    merged_data['Time_difference'] = merged_data['Time_difference'].round()

    merged_data['Status'] = ''

    merged_data.loc[merged_data['Time_difference']<360,'Status'] = 'Half-Day'

    merged_data.loc[merged_data['Time_difference']>=360,'Status'] = 'Full-Day'

    merged_data.loc[merged_data['Time_difference']<180 ,'Status'] = 'ERROR'

    merged_data.to_csv(f'Attendance\\final_attendance\\Attendance-{today_date}.csv',index=False)



