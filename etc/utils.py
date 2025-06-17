"""general utility for the project"""

from time import time, sleep
import math
import scipy.sparse as sp
import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

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
        if self.__frame_limit:
            sleep_dur = 1 / self.__frame_limit - (t - self.__last_frame_start_time)
            sleep(max(0, sleep_dur))
        self.__last_frame_start_time = t

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
    [```source```](https://github.com/tkipf/pygcn/tree/master/pygcn);\n
    
    I don't use torch.spmm (unlike the repo does), because the adj-matrix is very small (21x21) and the additional overhead might slow the operation down"""

    def __init__(self, in_features,out_features,bias=True):
        super(GraphConvolution,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias",None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)
    
    def forward(self,input,adj):
        support = torch.mm(input,self.weight)
        output = torch.spmm(adj,support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class TemporalConvolution(Module):
    def __init__(self):
        super(TemporalConvolution,self).__init__()

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
