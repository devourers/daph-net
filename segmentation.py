import torch
import torch.nn as nn


class SegmConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels


    def forward(self, X):
        return X


class DaphNet(nn.Module):
    '''
    TODO
    Segmentation daphnia model 
    '''
    def __init__(self, in_channels, out_channels) -> None:
        super(DaphNet, self).__init__()
        self.segconv1 = SegmConv2d(in_channels, out_channels)
        self.segconv2 = SegmConv2d(out_channels, out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return X

    def predict(self, X):
        with torch.no_grad():
            return self.forward(X)


def process_image_sequence(model, sequence):
    res = []
    for image in sequence:
        res.append(model.predict(image))
    return res


class SegmLoss(nn.Module):
    '''     
    TODO
    Loss counted between mask and predicition. 
    IoU or smth else.
    ''' 
    def __init__(self, a):
        super(self, SegmLoss).__init__()
    

    def forward(self, X, Y):
        return X