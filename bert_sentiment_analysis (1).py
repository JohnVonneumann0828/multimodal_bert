from bert import MAG_BertForSequenceClassification
from transformers import AutoTokenizer
import multimodal_driver
import torch
from global_configs import *
import inceptionv3
import librosa

#!python multimodal_driver.py --model bert-base-uncased

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#DEVICE=torch.device("cuda:0")
#change line 14 of bert.py to from transformers.activations import gelu, gelu_new, delete line 67
#change line 18 to from transformers.pytorch_utils import (  # noqa: F401

class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob
multimodal_config = MultimodalConfig(beta_shift=1e-3, dropout_prob=0.5)
print("stuff")
model = MAG_BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', multimodal_config=multimodal_config, num_labels=1,
    )
model.to(DEVICE)
#model.load_state_dict(torch.load("/content/gdrive/MyDrive/MAG_Model_Result/model.pth")["model_state_dict"])
print("stuff")
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
print("End!")
for i in range(len(data)):
  print(outputs[i])
  print(data[i][1])
