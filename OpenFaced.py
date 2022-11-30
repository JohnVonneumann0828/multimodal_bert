import sys
#from paths import SDK_PATH, DATA_PATH #, WORD_EMB_PATH, CACHE_PATH
SDK_PATH=r"E:\School\Project\OpenSmile"
DATA_PATH=r"E:\School\Project\OpenSmile\data"
if SDK_PATH is None:
    print("SDK path is not specified! Please specify first in constants/paths.py")
    exit(0)
else:
    sys.path.append(SDK_PATH)
#!pip install validators
#!pip install colorama
# !pip install cmu-multimodal-sdk
import colorama
import h5py
import hashlib
import validators
import json
import os
import re
import numpy as np
import mmsdk
from mmsdk import mmdatasdk as md
from subprocess import check_call, CalledProcessError

# # create folders for storing the data
#if not os.path.exists(DATA_PATH):
#    check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

# # download highlevel features, low-level (raw) data and labels for the dataset MOSI
# # if the files are already present, instead of downloading it you just load it yourself.
# # here we use CMU_MOSI dataset as example.

DATASET = md.cmu_mosei
print(DATASET)

#print('DATAPATH', DATA_PATH)
#md.mmdataset(DATASET.highlevel, DATA_PATH)

a=md.mmdataset(DATASET.highlevel, DATA_PATH)
    #print("High-level features have been downloaded previously.")

#try:
#    mydataset=md.mmdataset(DATASET.raw, DATA_PATH)
#except RuntimeError:
#    print("Raw data have been downloaded previously.")
    
# try:
#     md.mmdataset(DATASET.labels, DATA_PATH)
# except RuntimeError:
#     print("Labels have been downloaded previously.")
