import os
import torch.nn as nn
import torch.nn.functional as f
from etc.utils import GraphConvolution, convert_adjacency_matrix

DATA_DIR = os.path.join(os.getcwd(),"training/data")
GESTURE_COUNT = 3
SKELETON = [
             (0,1),(0,5),(0,9),(0,13),(0,17),(5,9),(9,13),(13,17), # palm
             (1,2),(2,3),(3,4), #thumb
             (5,6),(6,7),(7,8), #index finger
             (9,10),(10,11),(11,12), #middle finger
             (13,14),(14,15),(15,16), #ring finger
             (17,18),(18,19),(19,20) #pinky
             ]

class ST_GCN(nn.Module):
    '''ST-GCN (Spatial-Temporal Graph Convolutional Network)\n\nI assume T=50: 5s @ 50fps'''
    def __init__(self, n_classes):
        super(ST_GCN,self).__init__()

        ################## conf ##################
        self.adj = convert_adjacency_matrix(SKELETON)
        self.dropout = nn.Dropout(p=0.1)

        ################# layers #################
        # input shape: (B,T=50,P=21,D=3)
        self.sgc1 = GraphConvolution(3,8)
        # input shape: (B,T=50,P=21,D=8)
        self.tc1 = nn.Conv2d(in_channels=50,out_channels=128,kernel_size=1,stride=1,padding=0) # 1x1 temporal convolution
        # input shape: (B,T=50,P=21,D=8)
        self.sgc2 = GraphConvolution(8,16)
        # input shape: (B,T=128,P=21,D=16)
        self.tc2 =  nn.Conv2d(in_channels=128,out_channels=256,kernel_size=1,stride=1,padding=0) # 1x1 temporal convolution
        # input shape: (B,T=256,P=21,D=16)
        self.pool = nn.AdaptiveAvgPool2d((21,16)) # output size: (P, D)
        self.fc = nn.Linear(21*16,n_classes)

    def forward(self,x):

        x = f.relu(self.sgc1(x,self.adj))
        x = f.relu(self.tc1(x))
        x = f.dropout(x,self.dropout,training=self.training)
        x = f.relu(self.sgc2(x,self.adj))
        x = f.relu(self.tc2(x))
        x = f.dropout(x,self.dropout,training=self.training)
        x = self.pool(x)
        x = self.fc(x)

        return x


model = ST_GCN(GESTURE_COUNT)

