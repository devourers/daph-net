import segmentation
import torch
import os
import numpy as np
import PIL
import tqdm
import random
import matplotlib.pyplot as plt 


def load_data(path):
    '''
    Load images and masks. 
    '''
    files = os.listdir(path +'/masks')
    images = []
    masks = []
    print('Loading images and masks from \'' + path + '\' ... ')
    for file in tqdm.tqdm(files):
        curr_mask = np.asarray(PIL.Image.open(path + '/masks/' + file))
        curr_frame = np.asarray(PIL.Image.open(path + '/frames/' + file))
        images.append(curr_frame)
        masks.append(curr_mask)
    return images, masks


def form_dataset(images, masks, test_ratio):
    '''
    Form dataset from masks and images.
    '''
    dataset = []
    for image, mask in zip(images, masks):
        dataset.append([np.array(image), np.array(mask)])
    test_size = int(len(dataset) * test_ratio)
    random.shuffle(dataset)
    test_dataset = dataset[:test_size]
    train_dataset = dataset[test_size:]
    return np.array(train_dataset), np.array(test_dataset)


def train(model, train_images, train_masks, test_images, test_masks, n_epochs, loss_function, model_path):
    '''
    TODO
    Train model with given data.
    '''
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    log = []
    print('Training model...')
    for i in tqdm.tqdm(range(n_epochs)):
        y_pred = model.forward(train_images)
        train_loss = loss_function(y_pred, train_masks)
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        test_pred = model.forward(test_images)
        test_loss = loss_function(test_pred, test_images)
        log.append(train_loss, test_loss)
    torch.save(model.state_dict(), model_path)
    return log


images, masks = load_data()
train_dataset, test_dataset = form_dataset(images, masks, 0.1)
f, arr = plt.subplots(2)
arr[0].imshow(train_dataset[10][0], cmap='gray')
arr[1].imshow(train_dataset[10][1], cmap='gray')
plt.show()