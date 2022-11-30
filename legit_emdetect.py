import dlib
import numpy
import cv2
from transformers import AutoTokenizer
from bert import MAG_BertForSequenceClassification
from transformers import AutoTokenizer
import multimodal_driver
import torch
from global_configs import *
import numpy as np
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import IPython
import os
import pyaudio

#Loading The Model
#Loading Bert
class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
multimodal_config = MultimodalConfig(beta_shift=1e-3, dropout_prob=0.5)
model = MAG_BertForSequenceClassification.from_pretrained(
                            'bert-base-uncased', multimodal_config=multimodal_config, num_labels=1,
                        )
#import pickle
#with open("D:\Epping_Boys_High_School\Project\mosei.pkl",'rb') as f:
#  data=pickle.load(f)['test']
#audio=data[0][0][2]
#word=data[0][0][0]
#print(word)
model.to(DEVICE)
model.load_state_dict(torch.load("E:\model.pth")["model_state_dict"])
EPS=0
#Loading Visual
#Load segment audio classification model

model_path = r"E:\audio_NN_New2022_11_21_10_15_06_acc_90.83.json"
model_name = r"E:\audio_NN_New2022_11_21_10_15_06_acc_90.83.h5"

# Model reconstruction from JSON file
with open(model_path + model_name + '.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(model_path + model_name + '.h5')

# Replicate label encoder
lb = LabelEncoder()
lb.fit_transform(['Calling', 'Clapping', 'Falling', 'Sweeping', 'WashingHand', 'WatchingTV','enteringExiting','other'])

#Some Utils

# Plot audio with zoomed in y axis
def plotAudio(output):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,10))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    ax.margins(2, -0.1)
    plt.show()

# Plot audio
def plotAudio2(output):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    plt.show()

def minMaxNormalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr-mn)/(mx-mn)

def redictSound(X):
    clip, index = librosa.effects.trim(X, top_db=20, frame_length=512, hop_length=64) # Empherically select top_db for every sample
    stfts = np.abs(librosa.stft(clip, n_fft=512, hop_length=256, win_length=512))
    stfts = np.mean(stfts,axis=1)
    stfts = minMaxNormalize(stfts)
    result = model.predict(np.array([stfts]))
    predictions = [np.argmax(y) for y in result]
    #print(lb.inverse_transform([predictions[0]])[0])
    plotAudio2(clip)

CHUNKSIZE = 22050 # fixed chunk size
RATE = 22050

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

#noise window
data = stream.read(10000)
noise_sample = np.frombuffer(data, dtype=np.float32)
print("Noise Sample")
plotAudio2(noise_sample)
loud_threshold = np.mean(np.abs(noise_sample)) * 10
print("Loud threshold", loud_threshold)
audio_buffer = []
near = 0


#Start Detecting
class Face_Emotion():
    def __init__(self):
        self.detector=dlib.get_frontal_face_detector()
        self.cap=cv2.VideoCapture(0)
        self.predictor=dlib.shape_predictor("a:\\shape_predictor_68_face_landmarks.dat")
        self.cap.set(3,480)
        flag,im_rd=self.cap.read()
        k=cv2.waitKey(1)
        img_gray=cv2.cvtColor(im_rd,cv2.COLOR_RGB2GRAY)
        dets=self.detector(img_gray,0)
        listx_1 = [28, 30, 29, 29, 28, 28, 28, 28, 27, 27, 28, 27, 29, 29, 29, 29, 28, 28, 27, 27, 28, 28, 28, 29, 30,30, 32, 32, 33, 33, 35, 36, 38, 39, 40, 38, 43, 43, 43, 44, 45, 46, 46, 47, 49, 51, 53, 52, 56, 59,60, 60, 63, 70, 73, 77, 74, 75, 81, 85]
        listy_1 = [12, 14, 13, 13, 12, 12, 12, 12, 12, 12, 12, 11, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 13, 14,14, 15, 15, 16, 16, 17, 17, 19, 19, 19, 19, 22, 22, 22, 22, 24, 25, 26, 26, 28, 26, 26, 26, 28, 28,30, 30, 30, 31, 31, 36, 44, 45, 48, 46]
        self.browfit=numpy.polyfit(listx_1,listy_1,1)
        listx_2=[14, 14, 14, 14,15, 15, 15, 16, 16, 16, 16, 16, 17,17, 18,18, 18, 18, 20, 19, 20,21,22,25, 26, 26, 25, 25, 26, 26, 27, 28, 30,32, 33, 33, 36, 37, 39, 39, 40, 40, 41,48, 50, 52]
        listy_2=[11, 11, 11, 11, 14, 14, 14,16, 16, 16, 16, 16,17,17, 18,18, 18, 18, 19, 18, 19, 22,24,26, 26, 26, 26, 26, 26, 26, 28, 29, 31, 32, 35, 35, 37, 39, 40, 40, 42, 42, 45, 45, 45, 48]
        self.eyefit=numpy.polyfit(listx_2,listy_2,1)
        listx_3=[45, 46, 47, 48, 49, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 65, 66, 68, 69, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 86, 88, 91, 93, 94, 95, 97, 99, 100, 102, 103, 105, 107, 108, 109, 110, 112, 113, 117, 119, 122, 123, 127, 129, 131, 132, 135, 140, 141, 143, 147, 148, 151, 154, 157, 160, 168, 170, 172, 179, 182, 184, 200, 201, 203, 204, 213, 218]
        listy_3=[26, 27, 28, 29, 30, 31, 32, 34, 33, 35, 36, 37, 38, 40, 39, 41, 42, 44, 43, 45, 46, 47, 48, 50, 49, 51, 53, 54, 55, 58, 57, 56, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 73, 72, 74, 75, 76, 79, 83, 80, 81, 84, 85, 87, 88, 92, 93, 97, 100, 106, 108, 110, 115, 114, 116, 123, 124, 121, 126, 128, 133, 136, 140, 144, 149, 158, 167, 163]
        self.widthfit=numpy.polyfit(listx_3,listy_3,1)
        listx_4=[7.0,  8.0, 9.0,11.0, 12.0, 13.0, 14.0, 15.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 26.0, 31.0, 32.0, 33.0, 35.0, 39.0, 40.0, 41.0, 43.0, 46.0]
        listy_4=[10.0, 12.0, 13.0,15.0, 17.0, 18.0, 21.0,  23.0, 24.0, 27.0, 31.0, 33.0, 35.0, 37.0, 39.0, 42.0, 43.0,49.0, 53.0, 55.0, 58.0, 59.0, 63.0, 65.0]
        self.heightfit=numpy.polyfit(listx_4,listy_4,1)
    def learn_face(self):
        while (self.cap.isOpened()):
            self.smile=False
            self.amazed=False
            self.open=False
            self.angry=False
            font = cv2.FONT_HERSHEY_SIMPLEX
            flag,im_rd=self.cap.read()
            k=cv2.waitKey(1)
            img_gray=cv2.cvtColor(im_rd,cv2.COLOR_RGB2GRAY)
            dets=self.detector(img_gray,0)
            
            data = stream.read(CHUNKSIZE)
            current_window = np.frombuffer(data, dtype=np.float32)
            current_window = nr.reduce_noise(audio_clip=current_window, noise_clip=noise_sample, verbose=False)

            if(audio_buffer==[]):
                audio_buffer = current_window
            else:
                if(np.mean(np.abs(current_window))<loud_threshold):
                    print("Inside silence reign")
                    if(near<10):
                        audio_buffer = np.concatenate((audio_buffer,current_window))
                        near += 1
                    else:
                        predictSound(np.array(audio_buffer))
                        audio_buffer = []
                        #near
                else:
                    print("Inside loud reign")
                    near = 0
                    audio_buffer = np.concatenate((audio_buffer,current_window))
            if len(dets)!=0:
                for i in range(len(dets)):
                    for k,d in enumerate(dets):
                        cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (255,0,0))
                        shape=self.predictor(im_rd,d)
                        '''
                        visual=[]
                        for i in range(68):
                            visual.append(shape.part(i).x)
                        for i in range(68):
                            visual.append(shape.part(i).y)
                        visual.append(0)
                        visual=numpy.array(visual)
                        visual = numpy.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + numpy.std(visual, axis=0, keepdims=True)))
                        print(visual)
                        #audio=[[0 for i in range(74)]]
                        #print(audio.shape)
                        text=["Ha"]
                        audio=[0 for i in range(100)]
                        data=[[text,visual,audio]]
                        alist=multimodal_driver.convert_to_features(data,100,tokenizer)
                        #print(alist[0].input_ids.size())
                        input_ids=[]
                        visual=[]
                        acoustic=[]
                        attention_mask=[]
                        position_ids=[]
                        for i in alist: 
                            input_ids.append(i.input_ids)
                            visual.append(i.visual)
                            acoustic.append(i.acoustic)
                            attention_mask.append(i.input_mask)
                            position_ids.append(i.segment_ids)
                        input_ids=torch.tensor(input_ids).to(DEVICE)
                        visual=torch.tensor(visual).float().to(DEVICE)
                        acoustic=torch.tensor(acoustic).float().to(DEVICE)
                        attention_mask=torch.tensor(attention_mask).to(DEVICE)
                        position_ids=torch.tensor(position_ids).to(DEVICE)
                        outputs = model(input_ids, visual, acoustic, attention_mask, position_ids)
                        if(outputs[0]>0):
                            cv2.putText(im_rd, "Positive Emotion!", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 0, 255), 2, 4)
                        else:
                            cv2.putText(im_rd, "Negative!", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 0, 255), 2, 4)
                        '''
                        #Old Project!
                        mouth_width=shape.part(54).x-shape.part(48).x
                        mouth_height=shape.part(57).y-shape.part(51).y
                        eye_45=shape.part(46).y-shape.part(44).y
                        brow_dis=shape.part(21).x-shape.part(22).x
                        standard_width=shape.part(5).y-shape.part(0).y
                        standard_height=shape.part(30).y-shape.part(28).y
                        standard_eye=shape.part(47).x-shape.part(42).x
                        standard_brow=shape.part(42).x-shape.part(30).x
                        q=numpy.polyval(self.widthfit,standard_width)
                        w=numpy.polyval(self.heightfit,standard_height)
                        e=numpy.polyval(self.eyefit,standard_eye)
                        r=numpy.polyval(self.browfit,standard_brow)
                        if mouth_width>q:
                            self.smile=True
                        if mouth_height>w:
                            self.amazed=True
                        if eye_45>e:
                            self.open=True
                        if brow_dis<r:
                            self.angry=True
                            
                        if self.smile:
                            if self.open:
                                cv2.putText(im_rd, "Amazed ", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 0, 255), 2, 4)
                            elif self.amazed:
                                cv2.putText(im_rd,"LOL", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 0, 255), 2,4)
                            else:
                                cv2.putText(im_rd,"Happy", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 0, 255), 2, 4)
                        else:
                            if self.amazed:
                                cv2.putText(im_rd,"amazed", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 0, 255), 2, 4)
                            else:
                                cv2.putText(im_rd,"Natural", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 0, 255), 2, 4)
                             
                                                
            else:
                cv2.putText(im_rd, "No FaceQWQ", (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("camera",im_rd)
        self.cap.release()
        cv2.destroyAllWindows()
if __name__=="__main__":
    face=Face_Emotion()
    face.learn_face()
                
                            
            
            
        
        
        
