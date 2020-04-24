# numpy for converting tensors to feed in to matplotlib
import numpy as np

# pytorch for all the deep learning magic
import torch
# nn module for neural network layers and optim module for optimizers
from torch import nn, optim
# torchvision's datasets for importing our dataset, models for pretrained convnets and transforms for preprocessing our dataset
from torchvision import datasets, models, transforms
# OrderedDict for specifying our network to nn.Sequential
from collections import OrderedDict
# time for timing things
import time
# json for importing json files
import json
# argparse for parsing arguments
import argparse
# Image from Python Image Library for processing image files
from PIL import Image

def main():
    # get the inputed arguments or their defaults with the get_input_args() function
    input_args = get_input_args()
    
    # open category to name json
    with open(input_args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    device = use_gpu(input_args.gpu)
    
    # get probabilities and classes
    top_probs, top_classes = predict(input_args.image, input_args.checkpoint, input_args.top_k, device)
    
    top_flowers = []
    
    for top_class in top_classes:
        top_flowers.append(cat_to_name[top_class])
    
    print("The classifier predictions are:\n")
    leaderboard_count = 0
    for flower, prob in zip(top_flowers, top_probs):
        leaderboard_count += 1
        print(f"{leaderboard_count}. {flower.capitalize()}, probability: {prob:.3f}")
        
    

def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    the program is run from a terminal. Default arguments are parsed if the user
    does not specify them.
    Command Line Arguments:
      1. Path to image for classification as image with no default
      2. Path to checkpoint file for classification network as checkpoint with no default
      3. Top k number of predictions to print as top_k with default value of 5
      4. Path to category names json file as category_names with default of 'cat_to_name.json'
      5. Inference on GPU as --gpu if flag parsed
    This function returns these arguments as an ArgumentParser object.
    Parameters:
      None - simply using argparse module to create and store command line arguments
    Returns:
      parse_args() - data structure that stores the command line arguments object
    """
    # create parser object with ArgumentParser
    parser = argparse.ArgumentParser()
    
    # add the command line arguments to the parser object using add_argument() method
    # image path
    parser.add_argument('image', action = "store", type = str, help = 'Path to image for classification')
    # checkpoint path
    parser.add_argument('checkpoint', action = "store", type = str, help = 'Path to checkpoint file for classification network')
    # top_k number of predictions
    parser.add_argument('--top_k', type = int, default = 5, help = 'Top k number of predictions to print')
    # category names json file path
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'Path to category names json file')
    # train on gpu
    parser.add_argument('--gpu', action = "store_true", default = False, help = 'CUDA-capable GPU is attempted to be used if this argument is present')
    
    # return the collection of arguments
    return parser.parse_args()


def rebuild_model(checkpoint, device, frozen_features=True):
    """
    Accepts a path to a checkpoint file and returns a network.
    """
    
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    
    if model.features == models.densenet161:
        print("WE'VE GOT A DENSENET!")
    
    if frozen_features:
        for param in model.parameters():
            param.requires_grad = False
        print(f"\nModel loaded and features frozen.\n")
    else: 
        print(f"\nModel loaded with unfrozen features.\n"
              f"WARNING: The feature detection network will also be trained!\n")
    
    try:
        model.classifier
    except:
        print("WARNING: model.classifier didn't exist. Unknown feature network!")
        print("Exiting program.")
        exit()

    model.classifier = checkpoint['classifier']

    print(f"Pre-trained ConvNet classifier replaced with classifier trained for flower classification.\n"
          f"This classifier has been trained for {checkpoint['epochs_completed']} epochs and achieves:\n"
          f"Test loss: {checkpoint['test_loss']:.3f}.. "
          f"Test accuracy: {checkpoint['test_accuracy']:.3f}\n")
    
    # send network to device
    model.to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=0.5)   
    
    for word, word_string in zip((model, optimizer), ('model', 'optimizer')):
        word.load_state_dict(checkpoint[word_string+'_state_dict'])
        print(f"{word_string.capitalize()} state dict loaded!")

    model.class_to_idx = checkpoint['class_to_idx']
    print(f"Model class to index loaded!\n")
    
    epochs_completed = checkpoint['epochs_completed']
    
    
    return model, criterion, optimizer, scheduler


def process_image(image, width=256, height=256, sanity_check=False, debug=False):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # DONE: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img = img.resize((width, height))
    
    crop_width = 224 
    crop_height = 224
    left = (width - crop_width)/2 
    right = crop_width + left
    top = (height - crop_height)/2
    bottom = crop_height + top
    
    img = img.crop((left, top, right, bottom))
    
    if sanity_check == False:
        img = np.array(img)
    
        img = np.divide(img, 255)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (img - mean)/std
    
        img = img.transpose((-1, 0, 1))
    
        if debug == True:
            print(img.shape)
    
    return img

def predict(image_path, checkpoint, topk, device, debug=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model, criterion, optimizer, scheduler = rebuild_model(checkpoint, device, frozen_features=True)
   
    model.eval()    
    
    img = process_image(image_path)
    img = torch.from_numpy(img).float().to(device)
    # PyTorch expects a batch dimension: add this dimension and set to 1 for a single image
    img = img.view(1, 3, 224, 224)
    if debug == True:
        print(img)
    logps = model.forward(img)
    ps = torch.exp(logps)
    
    top_ps, top_idxs = ps.topk(topk, dim=1)
    
    class_to_idx = model.class_to_idx
    idx_to_class = dict(map(reversed, class_to_idx.items()))
    
    if debug == True:
        print(f"Indices to classes:\n {idx_to_class}")
    
    top_classes = []
    
    model.train()
    
    if debug == True:
        print(f"Top probabilities (tensor):\n {top_ps}\nTop indices (tensor):\n {top_idxs}")
    
    top_idxs = top_idxs.to('cpu').numpy()
    top_ps = top_ps.to('cpu').detach().numpy()
    
    top_ps = top_ps[0]
    
    if debug == True:
        print(f"Top indices (numpy):\n {top_idxs}")
    
    for idx in top_idxs[0]:
        top_classes.append(idx_to_class[idx])
    
    return top_ps, top_classes


def use_gpu(gpu):
    """
    Accepts a bool and enables GPU respectively, if a CUDA-capable GPU is available.
    Parameters:
      gpu - bool for whether GPU acceleration should be enabled or not
    Returns:
      device - torch.device('cuda') if gpu is True and a CUDA-capable GPU is available, otherwise torch.device('cpu')
    """
    
    if gpu == True:
        # use CUDA unless it's not available when sending tensors to 'device'
        cuda = torch.cuda.is_available()
        if cuda:
            print(f"\nUsing CUDA-capable GPU! Zoom, zoom!")
        else:
            print(f"\n--gpu argument was specified but CUDA-capable GPU was not available, reverting to CPU. Boo!")
        device = torch.device('cuda' if cuda else 'cpu')
    else:
        print(f"You're not using GPU acceleration. Consider the --gpu flag if you have a CUDA-capable GPU!")
        device = torch.device('cpu')
    return device


if __name__ == '__main__':
    main()