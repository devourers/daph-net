import os
import tqdm
import numpy as np
import PIL

def load_prediction_data(path):
    '''
    Load images and masks. 
    '''
    files = os.listdir(path)
    images = []
    print('Loading images and masks from \'' + path + '\' ... ')
    for file in tqdm.tqdm(files):
        curr_image = np.asarray(PIL.Image.open(path + '\\' +file))
        images.append(curr_image)
    return images