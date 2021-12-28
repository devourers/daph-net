
def divide_clip(path):
    raise NotImplementedError

def resize(image):
    raise NotImplementedError

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