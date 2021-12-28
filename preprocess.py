import torch
import torchvision

def divide_clip(path):
    raise NotImplementedError

def resize(image, size):
    res_image = torchvision.transforms.resize(image, size[0], size[1])
    return res_image

def form_image_sequence(images):
    res = []
    #and all other stuff like contrast etc.
    for image in images:
        res.append(resize(image))
    return res

def preprocess(path):
    loaded = divide_clip(path)
    preprocessed = form_image_sequence(loaded)
    return preprocessed