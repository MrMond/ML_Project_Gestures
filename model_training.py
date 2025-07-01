import os
import pickle
import tempfile
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.optim import Adam
from torch.utils.data import Dataset,DataLoader,random_split
from etc.utils import GraphConvolution, convert_adjacency_matrix, pickle_to_tensor

DATA_DIR = os.path.join(os.getcwd(),"training/skeleton_time_series")
MODEL_OUT_DIR = os.path.join(os.getcwd(),"models/classification")
GESTURE_COUNT = 3
SKELETON = [
             (0,1),(0,5),(0,9),(0,13),(0,17),(5,9),(9,13),(13,17), # palm
             (1,2),(2,3),(3,4), #thumb
             (5,6),(6,7),(7,8), #index finger
             (9,10),(10,11),(11,12), #middle finger
             (13,14),(14,15),(15,16), #ring finger
             (17,18),(18,19),(19,20) #pinky
             ]
LOOKUP = {"gesture_backward":0,"gesture_forward":1,"gesture_blacken":2}

# Define Model and Dataset

class ST_GCN(nn.Module):
    '''ST-GCN (Spatial-Temporal Graph Convolutional Network)\n\nI assume T=25: 5s @ 5fps'''
    def __init__(self, n_classes):
        super(ST_GCN,self).__init__()

        ################## conf ##################
        self.adj = convert_adjacency_matrix(SKELETON)
        self.dropout = nn.Dropout(p=0.1)

        ################# layers #################
        # input shape:  (B,D=3,T=25,P=21)
        self.sgc1 = GraphConvolution(3,8)
        # input shape: (B,D=8,T=25,P=21)
        self.tc1 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(25,1),stride=1,padding=0) # temporal convolution over timesteps
        # input shape: (B,D=16,T=25,P=21)
        self.sgc2 = GraphConvolution(16,32)
        # input shape: (B,D=32,T=25,P=21)
        self.tc2 =  nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(25,1),stride=1,padding=0) # temporal convolution over timesteps
        # input shape: (B,D=64,T=25,P=21)
        self.pool = nn.AdaptiveAvgPool2d((25,21)) # output size: (T, P)
        self.fc = nn.Linear(25*21,n_classes)

    def forward(self,x):
        
        print(f"got: {x.shape} expected: (B,D=3,T=25,P=21)")
        x = f.relu(self.sgc1(x,self.adj))
        print(f"got: {x.shape} expected: (B,D=8,T=25,P=21)")
        x = f.relu(self.tc1(x))
        print(f"got: {x.shape} expected: (B,D=16,T=25,P=21)")
        x = f.dropout(x,self.dropout,training=self.training)
        x = f.relu(self.sgc2(x,self.adj))
        print(f"got: {x.shape} expected: (B,D=32,T=25,P=21)")
        x = f.relu(self.tc2(x))
        x = f.dropout(x,self.dropout,training=self.training)
        print(f"got: {x.shape} expected: (B,D=64,T=25,P=21)")
        x = self.pool(x)
        print(f"got: {x.shape} expected: (T=25,P=21)")
        x = self.fc(x)
        print(f"returning: {x.shape}")

        return x

class PKL_Dataset(Dataset):
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.data = []
        self.labels = []
        # load in labels and path as pairs
        for root,dirs,_ in os.walk(data_dir):
            for dir in dirs:
                files = os.listdir(os.path.join(root,dir))
                self.data += files
                self.labels += [LOOKUP[dir]]*len(files)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        with open(os.path.join(self.data_dir,self.labels[idx],self.data[idx]),"rb") as of:
            dt = pickle.load(of)
        dt = pickle_to_tensor(dt)
        label = self.labels[idx]
        return dt,label

model = ST_GCN(GESTURE_COUNT)
dataset = PKL_Dataset(DATA_DIR)
trainset,valset = random_split(dataset,[0.8,0.2])
training_data = DataLoader(trainset,shuffle=True) 
validation_data = DataLoader(valset,shuffle=False) 

# Train Model

epochs = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

loop = tqdm(range(epochs),desc="loss = ___, val = ___")
losses = []
best_loss = float("inf")


model.train(True)
with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = os.path.join(tmpdir,"best_model.pth")
    for _ in loop:
        avg_loss = 0
        for data in training_data: # train 1 epoch
            inputs,labels = data
            optimizer.zero_grad() # zero gradients for each batch
            outputs = model(inputs)

            loss = loss_fn(outputs,labels)
            loss.backward()

            optimizer.step()

            avg_loss += loss.item()

        avg_loss = avg_loss / len(training_data)
        losses.append(avg_loss)

        model.eval()

        avg_vloss = 0
        with torch.no_grad(): # validate
            for data in validation_data:
                inputs,labels = data
                outputs = model(inputs)
                vloss = loss_fn(outputs,labels)
                avg_vloss += vloss.item()
            avg_vloss = avg_vloss/len(validation_data)
        loop.set_description(f"loss = {avg_loss:.5f} / val = {avg_vloss:.5f}")

        if best_loss > avg_vloss:
            best_loss = avg_vloss
            torch.save(model.state_dict(),tmp_path)
        else:
            model = ST_GCN(GESTURE_COUNT)
            model.load_state_dict(torch.load(tmp_path))

model.train(False)

torch.save(model.state_dict(),os.path.join(MODEL_OUT_DIR,"classifier.pth"))