#!/user.bin/python
# encoding: utf-8
import pygame
import sys
import traceback
import dlib
from pygame.locals import *
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
#import noisereduce as nr
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import IPython
import os
import pyaudio
from noisereduce.noisereducev1 import reduce_noise
import sounddevice as sd

pygame.init()
pygame.mixer.init()

global bg_size
bg_size = width, height = 1000, 500
background = pygame.display.set_mode(bg_size, RESIZABLE)

screen = pygame.image.load("image1/background.gif").convert()

pygame.display.set_caption("Poopie")
white = (255, 255, 255)
#game_music = pygame.mixer.music.load("F-777 - Deadlocked.mp3")
#pygame.mixer.music.set_volume(0.2)

fullscreen = False

GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
#WHITE = (255,255,255)
import pickle
with open("e:\\Project\\500-1500.pkl",'rb') as f:
    data1=pickle.load(f)['test']
#audio=data[0][0][2]
#word=data[1][0][1]
#@print(word)
#print(data[1][1])
class Train(pygame.sprite.Sprite):
    def __init__(self, bg_size):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load("image1/fighter2.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.width, self.height = bg_size[0], bg_size[1]
        self.rect.left, self.rect.top = (self.width - self.rect.width) / 2 + 300, self.height - self.rect.height
        self_mask = pygame.mask.from_surface(self.image)
        self.active = True
        self.hit_image = pygame.image.load("image1/fighter2_hit.png").convert_alpha()

    def hit(self):
        self.image = pygame.image.load("image1/1_hit.png")

    def move_right(self):
        self.speed = 7

        if self.rect.right < self.width:
            self.rect.left += self.speed

    def move_left(self):
        if self.rect.left > 0:
            self.speed = 5
            self.rect.left -= self.speed
        else:
            self.rect.left = 0


class ShockWave(pygame.sprite.Sprite):
    def __init__(self, position):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load("image1/shockwave.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = position
        self.speed = 12
        self.active = False
        self.mask = pygame.mask.from_surface(self.image)
        # self.width,self.height = bg_size[0],bg_size[1]

    def move(self):
        self.image = pygame.image.load("image1/shockwave.png")
        if self.rect.right < width:
            self.rect.left += self.speed
        else:
            self.active = False

    def move1(self):
        self.image = pygame.image.load("image1/shock_turn.png")
        if self.rect.right > 0:
            self.rect.left -= self.speed
        else:
            self.active = False

    def reset(self, position):
        self.rect.left, self.rect.top = position
        self.active = True


class EnemyWave(pygame.sprite.Sprite):
    def __init__(self, position):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load("image1/EnemyShock.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = position
        self.speed = 15
        self.active = False
        self.mask = pygame.mask.from_surface(self.image)
        # self.width,self.height = bg_size[0],bg_size[1]

    def move(self):
        self.rect.left -= self.speed

    def move1(self):
        self.rect.left += self.speed

    def reset(self, position):
        self.rect.left, self.rect.top = position
        self.active = True


class TurnWave(pygame.sprite.Sprite):
    def __init__(self, position):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load("image1/turn_wave.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = position
        self.speed = 8
        self.active = False
        self.mask = pygame.mask.from_surface(self.image)
        # self.width,self.height = bg_size[0],bg_size[1]

    def move1(self):
        self.rect.left += self.speed

    def reset(self, position):
        self.rect.left, self.rect.top = position
        self.active = True


class Fighter(pygame.sprite.Sprite):
    def __init__(self, bg_size):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load("image1/fighter1.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.speed = 7
        self.width, self.height = bg_size[0], bg_size[1]
        self.active = True
        self.rect.left, self.rect.top = (self.width - self.rect.width) / 2, self.height - self.rect.height
        self.rect.bottom = self.height
        self.mask = pygame.mask.from_surface(self.image)
        walk = []
        walk.extend([pygame.image.load("image1/fighter1.png").convert_alpha() \
                        , pygame.image.load("image1/walking.png").convert_alpha()])
        self.invincible = False

    # def move_up(self):
    #   self.speed=7
    #  if self.rect.top>0:
    #     self.rect.top -= self.speed
    # else:
    #   self.speed = 0
    #  self.rect.top+=self.speed
    # def move_down(self):
    ####else:
    #   self.rect.bottom = self.height
    def move_right(self):
        self.speed = 7
        if self.rect.right < self.width:
            self.rect.left += self.speed
        else:
            self.image = pygame.transform.flip(self.image, True, False)

    def move_left(self):
        if self.rect.left > 0:
            self.speed = 5
            self.rect.left -= self.speed
        else:
            self.rect.left = 0

    def stop(self):
        self.speed = 0

    def reset(self):
        self.active = True
        self.rect.bottom = self.height


def add_train(group, num):
    for i in range(num):
        t = Train(bg_size)
        group.add(t)


def m_hit(x, y):
    if x.rect.center == y.rect.center:
        return True


def add_me(group, num):
    for i in range(num):
        m = Fighter(bg_size)
        group.add(m)

def punch():

    PUNCH=USEREVENT
    pygame.time.set_timer(PUNCH, 1 * 750)
    
    if normal:
        hit = pygame.sprite.spritecollide(me, enemy, False, pygame.sprite.collide_mask)
        if hit:
            if attack_point < 200:
                attack_point += 5
                e_hp_point -= 20
            #if time_stop:
            #    e_hp_point -= 10
            #    attack_point += 20
def shock_wave():
    shock[shock_index].reset(me.rect.center)
    pygame.time.set_timer(CD, 2 * 1000)
    wave_wait = True
def initiate():
    global model
    global cap
    global predictor
    global lb
    global EPS
    global detector
    global stream
    global CHUNKSIZE
    global RATE
    global near
    global audio_buffer
    global noise_sample
    global loud_threshold
    global tokenizer
    global model1
    image=pygame.image.load("image1/Loading_page.png")
    background.blit(image,(0,0))
    pygame.display.flip()
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
    model.load_state_dict(torch.load("E:\model_11_27_13_18_highest.pth")["model_state_dict"])
    EPS=0
    #Loading Visual
    #Load segment audio classification model

    model_path = "e:\\project\\"
    model_name="audio_NN_New2022_11_21_10_15_06_acc_90.83"
    #model_name = r"e:\audio_NN_New2022_11_21_10_15_06_acc_90.83.h5"

    # Model reconstruction from JSON file
    with open(model_path + model_name + '.json', 'r') as f:
        model1 = model_from_json(f.read())

    # Load weights into the new model
    model1.load_weights(model_path + model_name + '.h5')

    # Replicate label encoder
    lb = LabelEncoder()
    lb.fit_transform(['Calling', 'Clapping', 'Falling', 'Sweeping', 'WashingHand', 'WatchingTV','enteringExiting','other'])

    detector=dlib.get_frontal_face_detector()
    cap=cv2.VideoCapture(0)
    predictor=dlib.shape_predictor("a:\\shape_predictor_68_face_landmarks.dat")
    cap.set(3,480)
    
    CHUNKSIZE = 8000 # fixed chunk size
    RATE = 8000

    # initialize portaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

    #noise window
    data = stream.read(10000)
    noise_sample = np.frombuffer(data, dtype=np.float32)
    print("Noise Sample")
    #plotAudio2(noise_sample)
    loud_threshold = np.mean(np.abs(noise_sample)) * 10
    print("Loud threshold", loud_threshold)
    audio_buffer = []
    near = 0

def minMaxNormalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr-mn)/(mx-mn)

def predictSound(X):
    clip, index = librosa.effects.trim(X, top_db=20, frame_length=512, hop_length=64) # Empherically select top_db for every sample
    stfts = np.abs(librosa.stft(clip, n_fft=512, hop_length=256, win_length=512))
    stfts = np.mean(stfts,axis=1)
    stfts = minMaxNormalize(stfts)
    result = model1.predict(np.array([stfts]))
    predictions = [np.argmax(y) for y in result]
    #print(lb.inverse_transform([predictions[0]])[0])
    #plotAudio2(clip)
    
def main():
    '''
    clock = pygame.time.Clock()

    pygame.mixer.music.play(-1)

    me1 = pygame.sprite.Group()
    add_me(me1, 1)

    hit = True

    overlap = False

    kick = True

    dec = False

    turn = False

    hitty = False

    punch = False

    Energy_bar = 0

    winning = False

    roundd = 3

    train = pygame.sprite.Group()
    add_train(train, 1)

    w_hit = USEREVENT

    JUMP = USEREVENT

    SPEC = USEREVENT + 3

    HURT = USEREVENT + 4

    delay = 100

    hp_point = 200

    attack_point = 0

    e_hp_point = 320

    hit_num = 0

    font = pygame.font.Font(None, 100)

    shock = []
    shock_index = 0
    SHOCK_NUM = 5
    for i in range(SHOCK_NUM):
        for each in me1:
            shock.append(ShockWave(each.rect.center))

    e_shock = []
    e_shock_index = 0
    E_SHOCK_NUM = 3
    for i in range(E_SHOCK_NUM):
        for each in train:
            e_shock.append(EnemyWave(each.rect.center))

    t_shock = []
    t_shock_index = 0
    T_SHOCK_NUM = 2
    for i in range(T_SHOCK_NUM):
        for each in train:
            t_shock.append(TurnWave(each.rect.center))

    CD = USEREVENT

    jump_vel = 0.0

    player_jumping = False
    jump_num = 0

    hit_num = 0

    jumpy = False

    jump_hit = 0

    close = False

    win_num = 0

    walk_index = 0

    WALK_NUM = 2

    switch_image = True

    heavy_punch = False

    heavy_kick = False

    kick_num = 0

    CD = USEREVENT

    wave_wait = False

    squat = False

    squat_punch = False

    hit_num_num = 0

    squat_kick = False

    heavy_kick_num = 0

    win_image = pygame.image.load("image1/turn_wave.png").convert_alpha()

    KICK = USEREVENT

    PUNCH = USEREVENT + 1

    E_CD = USEREVENT + 2
    e_wave = False

    heavy_punch_hurt = 0
    heavy_punch_num = 0

    INVINCIBLE_TIME = USEREVENT + 4

    move_with_boi = False

    jump_vel_e = 0.0
    jumping = False
    jump_num_e = 0

    me1_win = False

    hit_blit_count=0

    WIN_YAY = USEREVENT + 4

    bg_size = width, height = 1000, 500
    
    background = pygame.display.set_mode(bg_size, RESIZABLE)

    normal=True #主角是否为正常形态
    def jump():
        if jump_num < 1:
            jump_vel_e -= 24.0
            jumping = True
            jump_num_e += 1

    # def jump_e():
    #    f s.active and jump_num_e==0:
    # jump_vel_e-=30.0
    #              jumping=True
    #             jump_num_e+=1

    time_stop = False
    stop_num = 0
    '''
    #initiate
    clock = pygame.time.Clock()

    #pygame.mixer.music.play(-1)

    bg_size = width, height = 1000, 500
    
    background = pygame.display.set_mode(bg_size, RESIZABLE)

    initiate()
    #Characters
    global me
    global train
    global enemy 


    me=Fighter(bg_size)

    global shock
    global shock_index
    shock = []
    shock_index = 0
    SHOCK_NUM = 5
    for i in range(SHOCK_NUM):
        shock.append(ShockWave(me.rect.center))
    train=Train(bg_size)

    enemy=pygame.sprite.Group()
    enemy.add(train)

    #Variables
    global normal 
    speaking=False
    overlap = False
    turn=False
    normal=True

    global CD
    CD = USEREVENT
    font = pygame.font.Font(None, 100)
    PUNCH = USEREVENT

    audio_buffer = []

    global hp_point
    global e_hp_point
    global attack_point
    emotion=0
    attack_point = 0
    hp_point = 200
    e_hp_point = 320
    delay = 100
    near=0
    dets=[]
    while cap.isOpened():
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYUP:
                if event.key == [K_d]:
                    switch_image = True
            #elif event.type == KICK:
            #    heavy_kick_num = 0
            #elif event.type == PUNCH:
            #    heavy_punch_num = 0
            # if event.type == w_hit:
            #   for each in me1:
            #      each.image = pygame.image.load("image1/fighter1.png")
            #elif event.type == E_CD:
            #    e_wave = False
            #elif event.type == WIN_YAY:
            #    pygame.quit()
            #    sys.exit()


            # 全屏（F11）
            elif event.type == KEYDOWN:
                if event.key == K_F11:
                    global fullscreen
                    fullscreen = not fullscreen
                    if fullscreen:
                        pygame.display.set_mode((1920, 1080), FULLSCREEN | HWSURFACE)
                        for each in me1:
                            each.reset()
                    else:
                        pygame.display.set_mode(bg_size)
                #if event.key == K_END:
                #    pygame.quit()
                #    sys.exit()
            if event.type == VIDEORESIZE:
                bg_size = event.size
                width, height = event.size
                print(bg_size)
                background = pygame.display.set_mode(bg_size, RESIZABLE)
            elif event.type == CD:
                wave_wait = False
#            elif event.type == SPEC:
#                time_stop = False
        #视频和音频的数据读取
        background.fill(white)
        #音频
        duration = 0.083# seconds
        data = sd.rec(int(duration * RATE), samplerate=RATE, channels=2)
        print(len(data))
        current_window = np.frombuffer(data, dtype=np.float32)
        #current_window = reduce_noise(audio_clip=current_window, noise_clip=noise_sample, verbose=False)
        '''
        if(audio_buffer==[]):
            audio_buffer = current_window
        else:
            if(np.mean(np.abs(current_window))<loud_threshold):
                speaking=False
                emotion=0
                if(near<10):
                    audio_buffer = np.concatenate((audio_buffer,current_window))
                    near += 1
                else:
                    predictSound(np.array(audio_buffer))
                    audio_buffer = []
                        #near
            else:
                print("wow")
                flag,im_rd=cap.read()
                k=cv2.waitKey(1)
                img_gray=cv2.cvtColor(im_rd,cv2.COLOR_RGB2GRAY)
                dets=detector(img_gray,0)
                print(dets)
                speaking=True
                near = 0
                audio_buffer = np.concatenate((audio_buffer,current_window))
        '''
        speaking=True
        print(speaking)
        if len(dets)!=0 and speaking==True:
                print("lol")
                for i in range(len(dets)):
                    for k,d in enumerate(dets):
                        cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (255,0,0))
                        shape=predictor(im_rd,d)
                        visual=[]
                        text=[b'this' b'is' b'fantastic' b'four' b'the' b'first' b'one' b"it's"
 b'playing' b'here' b'in' b'the' b'screen' b'what' b'can' b'i' b'say'
 b'about' b'this' b'it' b'stars' b'jessica' b'alba' b'and' b'though'
 b'she' b'may' b'be' b'pretty' b"she's" b'not' b'the' b'best' b'actress'
 b'the' b'movie' b'is' b'just' b'really' b'not' b'that' b'great' b'uhh'
 b'the' b'acting' b'in' b'it' b'is' b'mediocre' b'the' b'special'
 b'effects' b'in' b'it' b'is' b'the' b'same' b'and' b'so' b'is' b'the'
 b'script' b'and' b'story' b'all' b'of' b"it's" b'mediocre' b'the'
 b'direction' b'was' b'was' b'stutter' b'horrible' b'i' b'was'
 b'completely' b'unsatisfied' b'by' b'this' b'movie' b'umm' b"there's"
 b'not' b'much' b'else' b'to' b'say' b'besides' b'that' b'but' b'the'
 b'story' b'was' b'so' b'bad' b'that' b'i' b'actually' b'considered'
 b'leaving' b'halfway' b'through' b'the' b'movie' b'so' b"don't" b'go'
 b'out' b'there' b'and' b'get' b'this' b'movie' b'please' b'because'
 b"it's" b'really' b'not' b'worth' b'it' b'if' b'you' b'wanna' b'see' b'a'
 b'good' b'superhero' b'if' b'you' b'wanna' b'see' b'a' b'good'
 b'superhero' b'movie' b'go' b'see' b'any' b'of' b'the' b'spider' b'mans'
 b'go' b'see' b'any' b'of' b'the' b'go' b'see' b'batman' b'begins'
 b"that's" b'a' b'good' b'one' b'or' b'go' b'see' b'any' b'of' b'the' b'x'
 b'men' b'those' b'are' b'all' b'great' b'movies' b'this' b'movie' b'just'
 b'jumped' b'on' b'the' b'superhero' b'bandwagon' b'and' b'just' b'did'
 b'not' b'do' b'a' b'good' b'job' b'at' b'retelling' b'the' b'true'
 b'story' b'of' b'the' b'fantastic' b'four' b'group' b'it' b'just'
 b"didn't" b'do' b'it' b'any' b'honor' b'i' b'would' b'be' b'ashamed'
 b'to' b'have' b'made' b'this' b'film' b'if' b'i' b'was' b'a' b'director'
 b'so' b'i' b"don't" b'suggest' b'you' b'go' b'out' b'there' b'to'
 b'watch' b'it' b'and' b'waste' b'your' b'money' b'on' b'this' b'thanks']

                        audio=[0 for i in range(100)]
                        for i in range(68):
                            if i != 49 and i!=55:
                                visual.append(0)
                                continue
                            visual.append(shape.part(i).x-d.left())
                        for i in range(68):
                            if i != 49 and i!=55:
                                visual.append(0)
                                continue
                            visual.append(shape.part(i).y-d.bottom())
                        visual.append(visual[-1])
                        #visual
                        visual=data1[1][0][1]
                        #visual=numpy.array(visual)
                        #visual = numpy.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + numpy.std(visual, axis=0, keepdims=True)))
                        #visual1=[]
                        #for i in range(len(text)):
                        #    visual1.append(visual)
                        #visual=numpy.array(visual1)
                        print(visual)
                        #audio=[[0 for i in range(74)]]
                        #print(audio.shape)
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
                        print(outputs)
                        if(outputs[0]>0):
                            emotion=1
                        else:
                            emotion=-1
        # 获取键盘操作
        print(emotion)
        key_pressed = pygame.key.get_pressed()
        # 跳跃
        #for each in me1:
        if 1:
            #if key_pressed[K_w] or key_pressed[K_SPACE]:
            #    jump()
            #    if jump_num < 1:
            #        jump_vel -= 24.0
            #        player_jumping = True
            #        jump_num += 1
            #if key_pressed[K_s] and not player_jumping:
            #    squat()
            #    #squat = True
            #elif not key_pressed[K_s]:
            #    squat = False
            # shock_wave
            #if key_pressed[K_r] and not wave_wait and not squat:
            if emotion==1:
                shock_wave()
                #shock[shock_index].reset(each.rect.center)
                #pygame.time.set_timer(CD, 2 * 1000)
                wave_wait = True

                # Jeffrey sucks
            elif emotion==-1:
                #if key_pressed[K_p] and not squat and punched==False:
                punch()
                #elif key_pressed[K_p] and squat:
                #    squat_punch = True
            '''
            # Normal Kick
            if player_jumping:
                if key_pressed[K_b]:
                jumpy = True

                ii_hit = pygame.sprite.spritecollide(each, train, False, pygame.sprite.collide_mask)
                if jump_hit == 0:
                    if ii_hit:
                        if attack_point < 200:
                            attack_point += 30
                        e_hp_point -= 20
                        jump_hit += 1
            else:
                if key_pressed[K_b]:
                if not (delay % 40):
                    kick = not kick
                    if not kick:
                        gg_hit = pygame.sprite.spritecollide(each, train, False, pygame.sprite.collide_mask)
                        each.image = pygame.image.load("image1/kick.png")
                        if gg_hit:
                            if attack_point < 200:
                                attack_point += 30
                            e_hp_point -= 35
            # jump_kick
            elif not player_jumping and jumpy:
                jumpy = False
                jump_hit = 0

            #elif player_jumping and key_pressed[K_t]:
            #    hitty = True
#
            #    oreifj_hit = pygame.sprite.spritecollide(each, train, False, pygame.sprite.collide_mask)
            #    if oreifj_hit and hit_num == 0:
            #        if attack_point < 200:
            #            attack_point += 5
            #        e_hp_point -= 15
            #        hit_num += 1
            #elif not player_jumping and hitty:
            #    hitty = False
            #    hit_num = 0

            if key_pressed[K_p]:
                if key_pressed[K_p] and not squat and punched==False:
                    heavy_punch_num += 1
                    pygame.time.set_timer(PUNCH, 1 * 750)
                    heavy_punch = True
                    if heavy_punch and heavy_punch_hurt == 0:
                        oioi_hit = pygame.sprite.spritecollide(each, train, False, pygame.sprite.collide_mask)
                        if oioi_hit:
                            if attack_point < 200:
                                attack_point += 5
                            e_hp_point -= 20
                            if time_stop:
                                e_hp_point -= 10
                                attack_point += 20
                            heavy_punch_hurt += 1
                elif key_pressed[K_p] and squat:
                    squat_punch = True
            elif not key_pressed[K_p]:
                squat_punch = False
                heavy_punch = False
                heavy_punch_hurt = 0

            if key_pressed[K_j] and not squat and heavy_kick_num == 0:
                heavy_kick = True
                heavy_kick_num += 1
                pygame.time.set_timer(KICK, 1 * 1000)
                if heavy_kick:
                    ioio_hit = pygame.sprite.spritecollide(each, train, False, pygame.sprite.collide_mask)
                    if ioio_hit and kick_num == 0:
                        kick_num += 1
                        if attack_point < 200:
                            attack_point += 30
                        e_hp_point -= 20
            elif not key_pressed[K_j] and not squat:
                heavy_kick = False
                kick_num = 0

            # squat and kick

            elif key_pressed[K_j] and squat:
                squat_kick = True
            elif not key_pressed[K_j] and squat == True:
                squat_kick = False
                ii_hit = pygame.sprite.spritecollide(each, train, False, pygame.sprite.collide_mask)
                if jump_hit == 0:
                    if ii_hit:
                        if attack_point <= 200:
                            attack_point += 30
                        e_hp_point -= 35
                        jump_hit += 1
            elif not player_jumping:
                jumpy = False
                jump_hit = 0

            if key_pressed[K_k]:
                each.reset()

            # specil attack

            if key_pressed[K_g] and attack_point >= 200:
                time_stop = True
                stop_num += 1
                attack_point = 0
                for target in train:
                    target.rect.right = width - each.rect.width
                    each.rect.right = target.rect.left
                pygame.time.set_timer(SPEC, 4 * 1000)

            if not turn:
                if key_pressed[K_d] and not overlap:
                    each.move_right()
                    for yay in train:
                        yay.move_left()
                    if not (delay % 10):
                        switch_image = not switch_image
                elif key_pressed[K_a]:
                    each.move_left()
                    for e in e_shock:
                        if (e.rect.left + e.rect.width) - (each.rect.left + each.rect.width) <= 100:
                            each.image = pygame.image.load("image1/defense.png").convert_alpha()
                            dec = True
                            ef_hit = pygame.sprite.spritecollide(e, me1, False, pygame.sprite.collide_mask)
                        elif (e.rect.left + e.rect.width) - (each.rect.left + each.rect.width) >= 100:
                            each.image = pygame.image.load("image1/fighter1.png").convert_alpha()
                            dec = False
                    for yay in train:
                        yay.move_right()
            elif turn:
                if key_pressed[K_d]:
                    each.move_right()
                    for yay in train:
                        yay.move_left()
                elif key_pressed[K_a] and not overlap:
                    each.move_left()
                    for e in e_shock:
                        if (e.rect.left + e.rect.width) - (each.rect.left + each.rect.width) <= 100:
                            each.image = pygame.image.load("image1/defense.png").convert_alpha()
                            dec = True
                            ef_hit = pygame.sprite.spritecollide(e, me1, False, pygame.sprite.collide_mask)
                        elif (e.rect.left + e.rect.width) - (each.rect.left + each.rect.width) >= 100:
                            each.image = pygame.image.load("image1/fighter1.png").convert_alpha()
                            dec = False
                    for yay in train:
                        yay.move_right()
                each.image = pygame.transform.flip(each.image, True, True)
            if move_with_boi:
                if key_pressed[K_d]:
                    for target in train:
                        each.move_right()
                        target.move_left()

            # 跳跃
            if player_jumping:
                for each in me1:
                    each.rect.top += jump_vel
                    jump_vel += 1.0

                    if each.rect.top == (height - each.rect.height) and not overlap:
                        player_jumping = False
                    elif overlap:
                        player_jumping = False

                    if not player_jumping and overlap:
                        player_jumping = True
                        for target in train:
                            if each.rect.left > 0:
                                each.rect.right = target.rect.left
                            else:
                                each.rect.left = target.rect.width
                    if each.rect.bottom >= height:
                        player_jumping = False

            if not player_jumping:
                for each in me1:
                    each.stop()
                    jump_num = 0
                    jump_vel = 0

            # 发射地方子弹
            if not (delay % 50) and not overlap and not e_wave:
                for each in train:
                    e_shock[e_shock_index].reset(each.rect.center)
                    e_shock_index = (e_shock_index + 1) % E_SHOCK_NUM
                pygame.time.set_timer(E_CD, 2 * 1000)
                e_wave = True
            if attack_point >= 1000:
                background.blit(screen, (0, 0))
            background.fill(white)
            '''

        # 我方子弹
        for s in shock:
            if s.active:
                if not turn:
                    s.move()
                    background.blit(s.image, s.rect)
                    hit = pygame.sprite.spritecollide(s, enemy, False, pygame.sprite.collide_mask)
                    if hit:
                        if attack_point < 200:
                            attack_point += 10
                        e_hp_point -= 10
                        s.active = False
                elif turn:
                    s.move1()
                    background.blit(s.image, s.rect)
                    hit = pygame.sprite.spritecollide(s, train, False, pygame.sprite.collide_mask)
                    if hit:
                        if attack_point < 200:
                            attack_point += 10
                        e_hp_point -= 10
                        s.active = False


        # 绘制血槽
        
        energy = hp_point / 200
        e_energy = e_hp_point / 400
        pygame.draw.line(background, BLACK, \
                             (0, 0), \
                             (width / 2, 0), \
                             100)

        pygame.draw.line(background, BLACK, \
                             (0, 0), \
                             (width / 2 * energy, 0), \
                             100)
        score_text = font.render("Enemy: %s" % str(e_hp_point), True, RED)
        background.blit(score_text, (600, 20))
        #if not turn:
        #    if me.rect.right - train.rect.right <= 300 and not overlap:
        #            me.move_left()

            # win!!!

            #spec = attack_point / 200
            #pygame.draw.line(background, BLACK, (0, 100), (200, 100), 70)

            #pygame.draw.line(background, (0, 0, 255), (0, 100), (200 * spec, 100), 70)

        # 如果主角死掉，游戏结束
        if hp_point <= 0:
            print("You've lost [sad]")
            pygame.quit()
            sys.exit()
        me.move_left()
        # 切换我方图片
        background.blit(me.image, me.rect)


        # for each in me1:
        #    for target in train:
        #        if each.rect.left-target.rect.left<=90:
        #            #jump hit
        #           jump_vel_e-=30.0
        #          jumping=True
        #         jump_num_e+=1
        #        target.rect.left-=2
        #       target.image=pygame.image.load("image1/fighter1.png")
        

        # background.blit(font.render(str('POINTS: %s'% str(attack_point)),True,(0,255,0)),(0,0))
        # background.blit(font.render(str('HP: %s'% str(hp_point)),True,(0,0,255)),(0,50))

        delay -= 1
        if not delay:
            delay = 100

        #if e_energy <= 0:
        #    me1_win = True
        #    pygame.time.set_timer(WIN_YAY, 2 * 10)
        #if me1_win:
        #    background.blit(pygame.image.load("image1/win_sign.png").convert_alpha(), (500, 0))
        pygame.display.flip()

        #if time_stop:
        #    clock.tick(30)
        #else:
        clock.tick(60)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
    except:
        traceback.print_exc()
        pygame.quit()
        input()

