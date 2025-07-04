"""general utility for the project"""

from time import time, sleep
import math
import scipy.sparse as sp
import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch import Tensor
import comtypes.client as pptx_client
import pyautogui

class FPS:
    """A class to be used with the ```with FPS() as fps:``` syntax to limit the framerate of a loop.
        \n**Usage**: Call ```fps.tick()``` at the end of the iteration
        \n**Attributes** (optional): limit:int
        \n**Properties** (read only): ```timestamp_ms:int```, ```measured_fps:int```"""

    def __init__(self, **kwargs):
        """A class to be used with the ```with FPS() as fps:``` syntax to limit the framerate of a loop.
        \n**Usage**: Call ```fps.tick()``` at the end of the iteration
        \n**Attributes** (optional): limit:int
        \n**Properties** (read only): ```timestamp_ms:int```, ```measured_fps:int```"""
        self.__measured_fps = 0
        self.__frame_list = []
        self.__start_time = 0
        self.__last_frame_start_time = 0
        self.__frame_limit = kwargs.get("limit")

    def __enter__(self):
        self.__start_time = time()
        self.__last_frame_start_time = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def tick(self) -> None:
        t = time()
        self.__frame_list.append(t)
        self.__frame_list = [ts for ts in self.__frame_list if ts + 1 > t]
        self.__measured_fps = len(self.__frame_list)
        self.__last_frame_start_time = t
        if self.__frame_limit:
            sleep_dur = 1 / self.__frame_limit - (t - self.__last_frame_start_time)
            sleep(max(0, sleep_dur))

    @property
    def measured_fps(self) -> int:
        return self.__measured_fps

    @measured_fps.setter
    def measured_fps(self, x):
        pass

    @property
    def timestamp_ms(self) -> int:
        return int((time() - self.__start_time) * 100)

    @timestamp_ms.setter
    def timestamp_ms(self, x):
        pass

class GraphConvolution(Module):
    """GCN Layer;\n
    modiefied version of this [```source```](https://github.com/tkipf/pygcn/tree/master/pygcn);\n
    
    I don't use torch.spmm (unlike the repo does), because the adj-matrix is very small (21x21) and the additional overhead might slow the operation down"""

    def __init__(self, in_features,out_features,adj_matrix_length=21,bias=True):
        super(GraphConvolution,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(out_features,in_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features,adj_matrix_length))
        else:
            self.register_parameter("bias",None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)
    
    def forward(self,input,adj):
        # expected input shape (B,D,T,P)
        B,D,T,P = input.shape
        output = []

        for t in range(T):
            x_t = input[:,:,t,:]
            x_t = torch.matmul(x_t,adj)
            x_t = torch.matmul(self.weight,x_t)
            if self.bias is not None:
                x_t = x_t + self.bias
            output.append(x_t)
        
        return torch.stack(output).permute(1,2,0,3)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def convert_adjacency_matrix(adjacency:list[tuple])->torch.Tensor:    
    rows,cols = zip(*adjacency)
    size = max(max(rows), max(cols)) + 1
    data = [1] * len(adjacency)

    adj = sp.coo_matrix((data,(rows,cols)),shape=(size,size))
    # make symmetrical
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # convert to tensor
    adj = adj.tocoo().astype(np.float32)
    idxs = torch.from_numpy(
        np.vstack((adj.row, adj.col)).astype(np.int64)
    )
    vals = torch.from_numpy(adj.data)
    shape = torch.Size(adj.shape)

    adj = torch.sparse.FloatTensor(idxs,vals,shape)

    return adj

def pickle_to_tensor(data:dict)->Tensor:
    output = []
    for timestep in data.values():
        landmarks = timestep.hand_landmarks[0] # disregard if 2 are found
        skeleton = []
        for point in landmarks:
            skeleton.append([point.x,point.y,point.z])
        output.append(skeleton)

    # permute to get correct shape (T,P,D)-->(D,T,P); unsqueeze to add Batch dimension B=1 (D,T,P)-->(B,D,T,P)
    return torch.as_tensor(output).permute(2,0,1)#.unsqueeze(0) 

class RotateTimeseries:
    def __init__(self):
        self._frames = []

    def continuous(self):
        '''check that the data has correct shape for tensor conversion'''
        return not(None in self._frames) and len(self._frames) == 30
    
    @property 
    def frames(self):
        return {i:f for i,f in enumerate(self._frames)}
    
    @frames.setter
    def frames(self,val):
        try: # assure that only "correct" skeleton results are in the array
            _ = val.hand_landmarks[0]
        except IndexError:
            val = None
        self._frames.append(val)
        if len(self._frames) > 30:
            self._frames = self._frames[-30:]

class PowerPoint:
    def __init__(self,path:str):
        self.client = None
        self.presentation = None
        self.load_powerpoint(path)

    def load_powerpoint(self,path:str):
        self.client = pptx_client.CreateObject("PowerPoint.Application")
        self.client.Visible = True

        pres = self.client.Presentations.open(path)
        self.presentation = pres.SlideShowSettings.Run()
    
    def advance_slide(self):
        if self.client:
            self.presentation.View.Next()

    def return_slide(self):
        if self.client:
            self.presentation.View.Previous()

    def toggle_blacken(self):
        if self.client:
            pyautogui.press('b') # toggle black screen

    # allow "with" syntax:

    def __enter__(self,path:str):
        self.load_powerpoint(path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.presentation.View.Exit()
        except:
            pass
        self.client = None
        self.presentation = None