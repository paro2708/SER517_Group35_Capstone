import torch
import torch.nn as nn

'''
This code defines a comprehensive model for gaze tracking, leveraging
both eye images and facial landmarks to predict the gaze direction.
'''
class openGaze(nn.Module):
    def __init__(self):
        super(openGaze, self).__init__()   
             
        self.eyeModel = eyeModel() #Initialized the eye model
        self.landMark = landMark() #Initialized landmark model

        #a sequential model to combine features from both eyes and landmarks and predict gaze direction
        self.combined_model = nn.Sequential(nn.Linear(512+512+16, 8),
                                            nn.BatchNorm1d(8, momentum=0.9),
                                            nn.Dropout(0.12),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(8, 4),
                                            nn.BatchNorm1d(4, momentum=0.9),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(4, 2),)  
        
    def forward(self, leftEye, rightEye, lms):
        #Flatten features of left and right eye
        l_eye_feat = torch.flatten(self.eyeModel(leftEye), 1)
        r_eye_feat = torch.flatten(self.eyeModel(rightEye), 1)
        
        lm_feat = self.landMark(lms) 
        
        #Concatenate the flattened features
        combined_feat = torch.cat((l_eye_feat, r_eye_feat, lm_feat), 1)
        out = self.combined_model(combined_feat) #Final output 
        return out
    
'''
This class defines a convolutional neural network (CNN) model for processing eye images.
It consists of three convolutional layers, each followed by batch normalization, a leaky
ReLU activation, average pooling, and dropout for regularization.
'''
class eyeModel(nn.Module): 
    def __init__(self):
        super(eyeModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=0), #First Convolutional Layer
            nn.BatchNorm2d(32, momentum=0.9),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.02),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0), #Second Convolutional Layer
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.02),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0), #Third Convolutional Layer
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.02),
        )

    def forward(self, x):
        x = self.model(x)
        return x

'''
This class defines a model for processing facial landmarks.
It uses fully connected layers with ReLU activations and batch normalization.
'''   
class landMark(nn.Module):
    def __init__(self):
        super(landMark, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 128),  #First fully connected Layer
            nn.BatchNorm1d(128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16), #Second fully connected Layer
            nn.BatchNorm1d(16, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16), #Third fully connected Layer
            nn.BatchNorm1d(16, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
    
xy = openGaze()
print(xy.eval())
