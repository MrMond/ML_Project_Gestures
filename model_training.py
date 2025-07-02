import os
import pickle
import tempfile
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transfomsv2
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from etc.utils import GraphConvolution, convert_adjacency_matrix, pickle_to_tensor

DATA_DIR = os.path.join(os.getcwd(), "training/skeleton_time_series")
MODEL_OUT_DIR = os.path.join(os.getcwd(), "models/classification")
GESTURE_COUNT = 3
SKELETON = [
    (0, 1),(0, 5),(0, 9),(0, 13),(0, 17),(5, 9),(9, 13),(13, 17),  # palm
    (1, 2),(2, 3),(3, 4),  # thumb
    (5, 6),(6, 7),(7, 8),  # index finger
    (9, 10),(10, 11),(11, 12),  # middle finger
    (13, 14),(14, 15),(15, 16),  # ring finger
    (17, 18),(18, 19),(19, 20),  # pinky
]
LOOKUP = {"gesture_backward": 0, "gesture_forward": 1, "gesture_blacken": 2}
REVERSE_LOOKUP = {i:k for k,i in LOOKUP.items()}

# Define Model and Dataset

class ST_GCN(nn.Module):
    """ST-GCN (Spatial-Temporal Graph Convolutional Network)\n\nI assume T=30: 3s @ 10fps"""

    def __init__(self, n_classes):
        super(ST_GCN, self).__init__()

        ################## conf ##################
        self.adj = convert_adjacency_matrix(SKELETON)
        self.dropout = 0.1

        ################# layers #################
        # input shape:  (B,D=3,T=30,P=21)
        self.sgc1 = GraphConvolution(3, 8)
        # input shape: (B,D=8,T=30,P=21)
        self.tc1 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(9,1), stride=1, padding=(4,0)
        )  # temporal convolution over timesteps
        # input shape: (B,D=16,T=30,P=21)
        self.sgc2 = GraphConvolution(16, 32)
        # input shape: (B,D=32,T=30,P=21)
        self.tc2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(9,1), stride=1, padding=(4,0)
        )  # temporal convolution over timesteps
        # input shape: (B,D=64,T=30,P=21)
        # fully connected layers to classify output
        self.fc1 = nn.Linear(64*30*21,30*21)
        self.fc2 = nn.Linear(30 * 21, n_classes)

    def forward(self, x):
        x = f.relu(self.sgc1(x, self.adj))
        x = f.relu(self.tc1(x))
        x = f.dropout(x, self.dropout, training=self.training)
        x = f.relu(self.sgc2(x, self.adj))
        x = f.relu(self.tc2(x))
        x = f.dropout(x, self.dropout, training=self.training)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class PKL_Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []
        self.labels = []
        # load in labels and path as pairs
        for root, dirs, _ in os.walk(data_dir):
            for dir in dirs:
                files = os.listdir(os.path.join(root, dir))
                self.data += files
                self.labels += [LOOKUP[dir]] * len(files)
        # I can use the torchvision transform module to add noise, because I have 3 dimensions (x,y,z)
        self.transform = transforms.Compose(
            [
                transfomsv2.GaussianNoise(
                    mean=0, sigma=0.025, clip=True
                )  # clip to keep values in range [0..1]
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with open(
            os.path.join(self.data_dir, REVERSE_LOOKUP[self.labels[idx]], self.data[idx]), "rb"
        ) as of:
            dt = pickle.load(of)
        try:
            dt = pickle_to_tensor(dt)
        except Exception as e:
            print(idx,self.labels[idx], self.data[idx])
            raise e
        dt = self.transform(dt)
        label = self.labels[idx]
        return dt, label


model = ST_GCN(GESTURE_COUNT)
dataset = PKL_Dataset(DATA_DIR)
trainset, valset = random_split(dataset, [0.8, 0.2])
training_data = DataLoader(trainset, shuffle=True)
validation_data = DataLoader(valset, shuffle=False)

# Train Model

epochs = 7
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

loop = tqdm(range(epochs), desc="loss = ___, val = ___")
losses = []
best_loss = float("inf")


model.train(True)
with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = os.path.join(tmpdir, "best_model.pth")
    for _ in loop:
        avg_loss = 0
        for data in training_data:  # train 1 epoch
            inputs, labels = data
            optimizer.zero_grad()  # zero gradients for each batch
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            avg_loss += loss.item()

        avg_loss = avg_loss / len(training_data)
        losses.append(avg_loss)

        model.eval()

        avg_vloss = 0
        with torch.no_grad():  # validate
            for data in validation_data:
                inputs, labels = data
                outputs = model(inputs)
                vloss = loss_fn(outputs, labels)
                avg_vloss += vloss.item()
            avg_vloss = avg_vloss / len(validation_data)
        loop.set_description(f"loss = {avg_loss:.5f} / val = {avg_vloss:.5f}")

        if best_loss > avg_vloss:
            best_loss = avg_vloss
            torch.save(model.state_dict(), tmp_path)
        else:
            model = ST_GCN(GESTURE_COUNT)
            model.load_state_dict(torch.load(tmp_path))

model.train(False)

torch.save(model.state_dict(), os.path.join(MODEL_OUT_DIR, "classifier.pth"))

with open(
    os.path.join(os.getcwd(), "training", f"run_losses_{time.time()}.pkl"), "wb"
) as of:
    pickle.dump(losses, of)
