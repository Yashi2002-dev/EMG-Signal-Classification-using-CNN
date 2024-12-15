import torch.nn as nn
import torch.nn.functional as F


class EMG2DClassifier_V0(nn.Module):

    def __init__(self, n_classes = 6, dropout = 0.15):

        super(EMG2DClassifier_V0, self).__init__()
        self.dropout = dropout
        #1 x 800 x 8
        self.Bn0 = nn.BatchNorm2d(1)
        self.Conv1 = nn.Conv2d(1, 32, kernel_size=(5,8), stride=(3,1), padding=(0,4))
        #32 x 266 x 9
        self.Bn1 = nn.BatchNorm2d(32)
        self.Conv2 = nn.Conv2d(32, 64, kernel_size=(9,5), stride=(3,1), padding=(1,1))
        self.Bn2 = nn.BatchNorm2d(64)
        #64 x 87 x 7        
        
        self.Conv3 = nn.Conv2d(64, 64, kernel_size=(7,5), stride=(3,1), padding=(1,2))
        #64 x 28 x 7
        self.Bn3 = nn.BatchNorm2d(64)
        self.Maxpool = nn.MaxPool2d(kernel_size=(7,6), stride=(3,1), padding=(0,3))
        #64 x 8 x 8 

        self.Conv4 = nn.Conv2d(64, 32, kernel_size=(5,5), stride=(2,2), padding=(1,1))
        #32 x 3 x 3 
        self.Bn4 = nn.BatchNorm2d(32)

        self.Conv5 = nn.Conv2d(32, 16, kernel_size=(3,3), stride=(1,1), padding=(0,0))
        #16 x 1 x 1
        self.Bn5 = nn.BatchNorm2d(16)
        self.FC_out = nn.Linear(16, n_classes)      
        
        
    def forward(self, x):        
        #print(x.shape)
        x = self.Bn0(x)
        x = self.Conv1(x)
        x = self.Bn1(x)
        x = F.relu(x)
        x = F.dropout(x,self.dropout)
        #print(x.shape)
        x = self.Conv2(x)
        x = self.Bn2(x)
        x = F.relu(x)
        x = F.dropout(x,self.dropout)
        #print(x.shape)
        x = self.Conv3(x)
        x = self.Bn3(x)
        x = F.relu(x)
        x = F.dropout(x,self.dropout)
        x = self.Maxpool(x)
        #print(x.shape)
        x = self.Conv4(x)
        x = self.Bn4(x)
        x = F.relu(x)
        x = F.dropout(x,self.dropout)
        #print(x.shape)
        x = self.Conv5(x)
        x = self.Bn5(x)
        x = F.relu(x)
        x = F.dropout(x,self.dropout)
        #print(x.shape)
        x = x.reshape(x.shape[0], -1)
        #print(x.shape)
        x = self.FC_out(x)
                       
        return x

    
    
class EMG2DClassifier_V1(nn.Module):

    def __init__(self, n_classes = 6, dropout = 0.15):

        super(EMG2DClassifier_V0, self).__init__()
        self.dropout = dropout
        #1 x 800 x 8
        self.Bn0 = nn.BatchNorm2d(1)
        self.Conv1 = nn.Conv2d(1, 32, kernel_size=(50,5), stride=(3,1), padding=(0,4))
        
        self.Bn1 = nn.BatchNorm2d(32)
        self.Conv2 = nn.Conv2d(32, 64, kernel_size=(9,5), stride=(3,1), padding=(3,2))
        self.Bn2 = nn.BatchNorm2d(64)
        
        self.Conv3 = nn.Conv2d(64, 64, kernel_size=(7,8), stride=(3,1), padding=(3,1))
        #64 x 28 x 7
        self.Bn3 = nn.BatchNorm2d(64)
        self.Maxpool = nn.MaxPool2d(kernel_size=(7,6), stride=(3,1), padding=(0,3))
        #64 x 8 x 8 

        self.Conv4 = nn.Conv2d(64, 32, kernel_size=(5,5), stride=(2,2), padding=(1,1))
        #32 x 3 x 3 
        self.Bn4 = nn.BatchNorm2d(32)

        self.Conv5 = nn.Conv2d(32, 16, kernel_size=(3,3), stride=(1,1), padding=(0,0))
        #16 x 1 x 1
        self.Bn5 = nn.BatchNorm2d(16)
        self.FC_out = nn.Linear(16, n_classes)      
        
        
    def forward(self, x):        
        #print(x.shape)
        x = self.Bn0(x)
        x = self.Conv1(x)
        x = self.Bn1(x)
        x = F.relu(x)
        x = F.dropout(x,self.dropout)
        #print(x.shape)
        x = self.Conv2(x)
        x = self.Bn2(x)
        x = F.relu(x)
        x = F.dropout(x,self.dropout)
        #print(x.shape)
        x = self.Conv3(x)
        x = self.Bn3(x)
        x = F.relu(x)
        x = F.dropout(x,self.dropout)
        x = self.Maxpool(x)
        #print(x.shape)
        x = self.Conv4(x)
        x = self.Bn4(x)
        x = F.relu(x)
        x = F.dropout(x,self.dropout)
        #print(x.shape)
        x = self.Conv5(x)
        x = self.Bn5(x)
        x = F.relu(x)
        x = F.dropout(x,self.dropout)
        #print(x.shape)
        x = x.reshape(x.shape[0], -1)
        #print(x.shape)
        x = self.FC_out(x)
                       
        return x    
    



    
class EMG2DClassifier_V2(nn.Module):

    def __init__(self, n_classes = 6, dropout = 0.5):

        super(EMG2DClassifier_V2, self).__init__()
        self.dropout = dropout
        #1 x 800 x 8
        self.Bn0 = nn.BatchNorm2d(1)
        self.Conv1 = nn.Conv2d(1, 32, kernel_size=(9,5), stride=(3,1), padding=(0,4))
        #64 x 264 x 12
        self.Bn1 = nn.BatchNorm2d(32)
        self.Conv2 = nn.Conv2d(32, 64, kernel_size=(7,7), stride=(2,1), padding=(0,3))
        self.Bn2 = nn.BatchNorm2d(64)
        #128 x 129 x 12        
        
        self.Conv3 = nn.Conv2d(64, 64, kernel_size=(7,7), stride=(2,1), padding=(0,2))
        #128 x 62 x 10
        self.Bn3 = nn.BatchNorm2d(64)
        #self.Maxpool = nn.MaxPool2d(kernel_size=(7,6), stride=(3,1), padding=(0,3))
        #Check dims after possible pool
         

        self.Conv4 = nn.Conv2d(64, 64, kernel_size=(7,7), stride=(2,1), padding=(0,2))
        #64 x 28 x 8 
        self.Bn4 = nn.BatchNorm2d(64)

        self.Conv5 = nn.Conv2d(64, 64, kernel_size=(5,5), stride=(2,1), padding=(0,2))
        #64 x 12 x 8
        self.Bn5 = nn.BatchNorm2d(64)
        
        self.Conv6 = nn.Conv2d(64, 32, kernel_size=(5,5), stride=(2,1), padding=(0,0))
        #32 x 4 x 4
        self.Bn6 = nn.BatchNorm2d(32)
        
        self.Conv7 = nn.Conv2d(32, 16, kernel_size=(4,4), stride=(1,1), padding=(0,0))
        #16 x 1 x 1
        self.Bn7 = nn.BatchNorm2d(16)
        
        self.FC_out = nn.Linear(16, n_classes)
        
    def forward(self, x):        
        #print(x.shape)        
        x = self.Bn0(x)
        x = self.Conv1(x)
        x = self.Bn1(x)
        x = F.relu(x)
        #x = F.dropout(x,self.dropout)
        #print(x.shape)
        x = self.Conv2(x)
        x = self.Bn2(x)
        x = F.relu(x)
        #x = F.dropout(x,self.dropout)
        #print(x.shape)
        x = self.Conv3(x)
        x = self.Bn3(x)
        x = F.relu(x)
        #x = F.dropout(x,self.dropout)
        #x = self.Maxpool(x)
        #print(x.shape)
        x = self.Conv4(x)
        x = self.Bn4(x)
        x = F.relu(x)
        #x = F.dropout(x,self.dropout)
        #print(x.shape)
        x = self.Conv5(x)
        x = self.Bn5(x)
        x = F.relu(x)
        
        x = self.Conv6(x)
        x = self.Bn6(x)
        x = F.relu(x)
        
        x = self.Conv7(x)
        x = self.Bn7(x)
        x = F.relu(x)
        #x = F.dropout(x,self.dropout)
        #print(x.shape)
        x = x.reshape(x.shape[0], -1)
        #print(x.shape)
        x = self.FC_out(x)
                       
        return x