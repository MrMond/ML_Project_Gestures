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

class Model(nn.Module):
    def __init__(self, nclass):
        super(Model,self).__init__()

        # conf
        self.adj = convert_adjacency_matrix(SKELETON)
        self.dropout = nn.Dropout(p=0.1)

        # layers
        self.sgc1 = GraphConvolution()
        self.tc1 = TemporalConvolution() # make this 1d convolution
        self.sgc2 = GraphConvolution()
        self.tc2 = TemporalConvolution() # make this 1d convolution
        self.pool = nn.AdaptiveAvgPool2d()
        self.fc = nn.Linear(,nclass)


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


model = Model(GESTURE_COUNT)

