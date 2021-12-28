import segmentation
import torch
import preprocess

def load_data(path):
    raise NotImplementedError

def form_dataset(data):
    raise NotImplementedError

def train(model, dataset, model_path=None):
    opt = torch.optim.Adam()
    raise NotImplementedError

def resume_training(model_path):
    raise NotImplementedError