import segmentation
import torch
import os
import numpy as np
import PIL
import tqdm


def load_data(path):
    '''
    Load images and masks. 
    '''
    files = os.listdir(path +'/masks')
    images = []
    masks = []
    for file in tqdm.tqdm(files):
        curr_mask = np.asarray(PIL.Image.open(path + '/masks/' + file))
        curr_frame = np.asarray(PIL.Image.open(path + '/frames/' + file))
        images.append(curr_frame)
        images.append(curr_mask)
    return images, masks

def form_dataset(images, masks):
    raise NotImplementedError

def train(model, dataset, n_epochs, loss_function, model_path=None):
    opt = torch.optim.Adam()
    raise NotImplementedError

def resume_training(model_path):
    raise NotImplementedError


print(load_data('W:/daphtrack/bin.x86_64-windows.release/testing_for_you'))