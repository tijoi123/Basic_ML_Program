from sys import platform
import numpy as np
import subprocess
import shutil
import ast
import sys
import os

import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import v2

import matplotlib.pyplot as plt

import copy
from torchvision import models
import torch.optim as optim
import time
from torch.optim import lr_scheduler

import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

detected_list = []
predsbering_list = []
img_name_list = []


# EVENT NAMES

combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"

''' 
	
	This function will load your trained model and classify the event from an image which is 
    sent as an input.
	
	'''
def classify_event(image):
    image = Image.open(image)

    model = torchvision.models.resnet152('IMAGENET1K_V2')
    for param in model.parameters():
        param.requires_grad = False


    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)# we do not specify ``weights``, i.e. create untrained model
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu'))) 
    model.eval()
# Apply transformations and convert it to a tensor
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        s=model(image)
    _, preds = torch.max(s, 1)
    if preds ==0:
        event = "combat"
    if preds ==3:
        event = "humanitarianaid"
    if preds ==4:
        event = "militaryvehicles"
    if preds ==2:
        event = "fire"
    if preds ==1:
        event = "destroyedbuilding"

    
    return event


def classification(img_name_list):
    for img_index in range(len(img_name_list)):
        img = "events/" + str(img_name_list[img_index]) + ".jpeg"
        detected_event = classify_event(img)
        print((img_index + 1), detected_event)
        if detected_event == combat:
            detected_list.append("combat")
        if detected_event == rehab:
            detected_list.append("rehab")
        if detected_event == military_vehicles:
            detected_list.append("militaryvehicles")
        if detected_event == fire:
            detected_list.append("fire")
        if detected_event == destroyed_building:
            detected_list.append("destroyedbuilding")
    shutil.rmtree('events')
    return detected_list

def detected_list_processing(detected_list):
    try:
        detected_events = open("detected_events.txt", "w")
        detected_events.writelines(str(detected_list))
    except Exception as e:
        print("Error: ", e)

def input_function():
    if platform == "win32":
        try:
            subprocess.run("input.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./input")
        except Exception as e:
            print("Error: ", e)
    img_names = open("image_names.txt", "r")
    img_name_str = img_names.read()

    img_name_list = ast.literal_eval(img_name_str)
    return img_name_list
    
def output_function():
    if platform == "win32":
        try:
            subprocess.run("output.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./output")
        except Exception as e:
            print("Error: ", e)


def main():
    ##### Input #####
    img_name_list = input_function()
    #################

    ##### Process #####
    detected_list = classification(img_name_list)
    detected_list_processing(detected_list)
    ###################

    ##### Output #####
    output_function()
    ##################

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        if os.path.exists('events'):
            shutil.rmtree('events')
        sys.exit()