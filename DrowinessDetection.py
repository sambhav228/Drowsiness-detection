import imutils
import face_recognition
import cv2
from scipy.spatial import distance as dist
import playsound
from threading import Thread
import numpy as np
import os
MIN_AER =0.30
EYE_AR_CONSEC_FRAMES =10
COUNTER = 0
ALARM_ON = False
def playAlarm(soundfile):
    playsound.playsound(soundfile)

def eye_aspect_ratio(eye):
    V1 = dist.euclidean(eye[1],eye[5])
    V2 = dist.euclidean(eye[2], eye[4])
    H = dist.euclidean(eye[0], eye[3])
    ear = (V1+V2)/(2.0*H)
    return ear

def main():
    global COUNTER,ALARM_ON
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3,320)
    video_capture.set(4,240)
    while True:
        ret ,frame = video_capture.read()
        face_landmarks_list = face_recognition.face_landmarks(frame)
        for face_landmark in face_landmarks_list:
            leftEye = face_landmark["left_eye"]
            rightEye = face_landmark["right_eye"]
            lpts = np.array(leftEye)
            rpts = np.array(rightEye)

            cv2.polylines(frame,[lpts],True,(255,255,0),1)
            cv2.polylines(frame, [rpts], True, (255, 255, 0), 1)

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR+rightEAR)/2

            if ear<MIN_AER:
                COUNTER+=1
                if COUNTER>=EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        t = Thread(target=playAlarm,args=('alarm.wav',))
                        t.daemon = True
                        t.start()
                    cv2.putText(frame,"ALERT! You are feeling asleep!",(10,10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
                    if COUNTER>=30:
                        shutDownComputer()
            else:
                COUNTER=0
                ALARM_ON = False
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (250, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.imshow("Sleep detection system",frame)
        if cv2.waitKey(1)== ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

