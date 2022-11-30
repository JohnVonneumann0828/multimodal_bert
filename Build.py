import numpy as np
import cv2
import librosa
#import recording # For now the recording feature
# Import system modules
import sys, string, os
#os.system("D:\Epping_Boys_High_School\Project\Code_Implementation\OpenFace_2.2.0_win_x64\OpenFaceOffline.exe")
cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# Define the codec and create VideoWriter object
out = cv2.VideoWriter('D:\Epping_Boys_High_School\Project\Code_Implementation\BERT_multimodal_transformer-master\output.avi', fourcc, 30.0, (640,480))
voice=True
def audio_extraction(f):
    rmse=librosa.feature.rms(f)
    y_harmonic=librosa.effects.harmonic(f,margin=1.0)
    pitch=librosa.core.pitch(f)
    
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True and voice:
        out.write(frame)
        
        #cv2.imshow('frame',frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    elif ret==True:
        if first:
            #run video using openface
            #run audio using librosa
            
        cv2.imshow('frame',frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break        
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
