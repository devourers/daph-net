import count
import segmentation
import json
import train
import torch
import torch.nn as nn
import numpy as np


def train_pipe(path, model_path, n_epochs, test_ratio):
    '''
    TODO
    Training pipeline for model.
    '''
    try:
        model = segmentation.Net()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print('Loaded model from \'' + model_path + '\' ...')
    except:
        model = segmentation.Net()
        print('No model was found, initializing new model...')

    images, masks = train.load_data(path)
    train_dataset, test_dataset = train.form_dataset(images, masks, test_ratio)

    train_images = torch.tensor(train_dataset[:, 0])
    train_masks = torch.tensor(train_dataset[:, 1])
    test_images = torch.tensor(test_dataset[:, 0])
    test_masks = torch.tensor(test_dataset[:, 1])

    loss_function = train.SegmLoss(1) #placeholder
    x_epochs = np.arange(0, n_epochs, 1)
    log = train.train(model, train_images, train_masks, test_images, test_masks, n_epochs, loss_function, model_path)
    plt.plot(x_epochs, log[:, 0], label = 'Train Loss')
    plt.plot(x_epochs, log[:, 1], label = 'Test Loss')
    plt.show()
    #test accuracy


def pred_pipe(path, model_path):
    raise NotImplementedError()


def main(path, model_path, real_data=None):
    print('i am not ready yet')