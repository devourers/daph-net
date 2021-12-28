import torch


class SegmConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        pass
    def forward(self, X):
        return X

class DaphNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DaphNet, self).__init__()

    def forward(self, X):
        return 0

    def predict(self, X):
        return 0


def train(model):
    raise NotImplementedError


def process_image_sequence(model, sequence):
    res = []
    for image in sequence:
        res.append(model.predict(image))

    return res