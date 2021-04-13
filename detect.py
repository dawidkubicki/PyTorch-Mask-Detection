import cv2
import dlib
import numpy as np
import torch
from torch import nn
import torchvision.transforms as tt
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from threading import Thread
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

flag_check=20
stats= ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
train_tfms = tt.Compose([tt.Resize((224, 224)),
                         #tt.RandomHorizontalFlip(),
                        tt.ToTensor(),
                        tt.Normalize(*stats)
                        ])



class ConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(54*54*16, 220)
        self.fc2 = nn.Linear(220, 100)
        self.fc3 = nn.Linear(100,2)
        
    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        
        return F.log_softmax(X, dim=1)





video = cv2.VideoCapture(0)
device = torch.device("cpu")
detector = dlib.get_frontal_face_detector() #Face detector obtained from OpenCV's dlib (Deep Learning Library)

no_mask_count = 0
five_second_count = 0

# def load_checkpoint(filepath):
#     checkpoint = torch.load(filepath, device)
#     model = checkpoint['model']
#     model.load_state_dict(checkpoint['state_dict'], strict=False)
#     for parameter in model.parameters():
#         parameter.requires_grad = False

#     model.eval()
#     return model

filepath = '/Users/dawidkubicki/Documents/models/entire_model.pt'
# loaded_model = load_checkpoint(filepath)


model = torch.load(filepath, map_location=torch.device('cpu'))
model.eval()

#Opens the video camera, sends each frame of the video to the model and obtains the prediction.This goes on till the 'Esc' key is pressed.

while True:
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x,y = face.left(), face.top()
        x1,y1 = face.right(), face.bottom()

        crop = frame[y:y1, x:x1]
        pil_image = Image.fromarray(crop, mode="RGB")
        pil=train_tfms(pil_image)
        img = pil.unsqueeze(0)

        yb = model(img)
        _, max  = torch.max(yb, dim=1)
        prediction = max.item()

#Puts a bounding box around the face with the corresponding text based on the prediction obtained

        if prediction == 1:
            cv2.putText(frame, "Wearing a Mask", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
            cv2.rectangle(frame, (x,y)  , (x1,y1), (0,255,0), 2)
            no_mask_count = 0

        elif prediction == 0:
            cv2.putText(frame, "Not Wearing a Mask", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)
            cv2.rectangle(frame, (x,y), (x1,y1), (255,0,0), 2)
            no_mask_count += 1

            if no_mask_count >= flag_check:
                if five_second_count==0:
                    print("no mask")
                no_mask_count = 0
                five_second_count += 1

                if five_second_count > 2:
                    print("no more mask")
                    continue


    cv2.imshow("Frame", frame)


    key = cv2.waitKey(1)
    if key == 27:
        break

video.release()
