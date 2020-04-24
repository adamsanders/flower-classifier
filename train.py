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

def main():
    # get the inputed arguments or their defaults with the get_input_args() function
    input_args = get_input_args()
    
    data_dir = input_args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # DONE: Define your transforms for the training, validation, and testing sets
    # RandomPerspective() commented out as it's not in 0.4.0
    train_transforms = transforms.Compose([#transforms.RandomPerspective(),
                                           transforms.RandomRotation(90),
                                           transforms.Resize(256),
                                           transforms.RandomResizedCrop(224, scale=(0.15,1.0)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(p=0.1), # in deployment we don't want the network to be useless if someone uploads an upside down picture!
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # DONE: Load the datasets with ImageFolder
    trainset = datasets.ImageFolder(train_dir, transform=train_transforms)
    validset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    testset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # DONE: Using the image datasets and the trainforms, define the dataloaders
    # define batch size as a variable to reuse in timing later
    batch_size = input_args.batch_size
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    
    num_batches = len(trainloader)
    
    # label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # select model architecture from command line argument
    if input_args.arch == 'densenet':
        model = models.densenet161(pretrained=True)
        print(f"\nPre-trained 'densenet-161' loaded as the feature detection network.")
    elif input_args.arch == 'vgg':
        model = models.vgg19_bn(pretrained=True)
        print(f"\nPre-trained 'vgg19_bn' loaded as the feature detection network.")
    else:
        print(f"\nWARNING: Invalid CNN architecture: '{input_args.arch}' was selected. Program exiting.")
        exit()
    
    # freeze all the params of the pretrained network
    for param in model.parameters():
        param.requires_grad = False
    
    model, device, criterion, optimizer, scheduler = classifier_setup(model,
                                                                      input_args.arch, 
                                                                      input_args.layers, 
                                                                      input_args.neurons,
                                                                      input_args.lr,
                                                                      input_args.epochs,
                                                                      input_args.patience,
                                                                      input_args.lr_factor,
                                                                      input_args.gpu,
                                                                      num_batches)
    
    print(f"\nValidation will be performed every {input_args.valid_every} epochs.")
    
    # verbose
    verbose = input_args.verbose
    
    # epochs completed (for resuming training)
    epochs_completed = 0
    
    # keep track of total time: training+validation
    total_epoch_time = 0

    # keep track of total training time
    total_train_time = 0

    # keep track of total validation time
    total_valid_time = 0

    # we'll use this later with modulo == 0 to report every this many steps
    report_multiplier = input_args.valid_every
    report_every = report_multiplier*len(trainloader) # report per report_multiplier number of epochs
    
    # total epochs to do
    total_epochs = epochs_completed + input_args.epochs

    # steps we've taken (for reporting loss per x steps)
    steps = 0
    # our running loss
    running_train_loss = 0

    print(f'\nNumber of batches in training set: {num_batches}...\n')
    # main training loop
    for epoch in range(input_args.epochs):
        epoch_time = 0
        train_time = 0
        valid_time = 0
        start = time.time()
        # loop over our batches in trainloader
        for inputs, labels in trainloader:
            # increment steps
            steps += 1
        
            # send inputs and labels to GPU if available
            inputs, labels = inputs.to(device), labels.to(device)
        
            # zero our gradients so they don't accumulate across batches
            optimizer.zero_grad()
        
            # get our log probabilities from forward pass through the network
            logps = model.forward(inputs)
            # calculate our training loss for this batch
            train_loss = criterion(logps, labels)
            # add the loss for this training batch to the running training loss
            running_train_loss += train_loss.item()
            # propagate the error backward through the network
            train_loss.backward()
            # if pytorch version <1.1.0, call scheduler here
            if optimizer_first() == False:
                scheduler.step(running_train_loss/len(trainloader))
            # take an optimizer step in the direction of the gradients multiplied by the learning rate
            optimizer.step()
            
            # we'll report every 'report_every' number of batches
            if steps % report_every == 0:
                valid_time = 0
                valid_start = time.time()
                # create running_valid_loss and accuracy within the scope of this
                running_valid_loss = 0
                accuracy = 0
                # set the model to evaluation mode to disable dropout
                model.eval()
                # stop tracking gradients for a speedup since we're not doing backprop here
                with torch.no_grad():
                    # loop over our batches in validloader
                    for inputs, labels in validloader:
                        # send inputs and labels to GPU if available
                        inputs, labels = inputs.to(device), labels.to(device)
                        # get log probabilities from a forward pass through the network
                        logps = model.forward(inputs)
                        # calculate the validation loss for this batch
                        valid_loss = criterion(logps, labels)
                    
                        # add the loss for this validation batch to the running validation loss
                        running_valid_loss += valid_loss.item()
                    
                        # calculate the accuracy for this batch
                        # probabilities from log probabilities
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        # if top_class and label matches set equals to True
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()            
                # scheduler here for newer versions of pytorch so patience can be based on validation rather
                # than training if wanted
                if optimizer_first() == True:
                    scheduler.step(running_train_loss/len(trainloader))
            
                valid_time = time.time() - valid_start
                if verbose:
                    print(f'\nDuration of this validation: {valid_time:.3f} seconds...')
            
                print(f'\nEpoch {1+epochs_completed}/{total_epochs}... '
                      f'Train loss: {running_train_loss/report_every:.3f}... '
                      f'Validation loss: {running_valid_loss/len(validloader):.3f}... '
                      f'Validation accuracy: {accuracy/len(validloader):.3f}...')
            
                model.train()
                running_train_loss = 0
            
        epoch_time = time.time() - start
        total_epoch_time += epoch_time
        
        total_valid_time += valid_time
        
        train_time = epoch_time - valid_time
        total_train_time += train_time
        
        epochs_completed += 1
        
        if verbose:
            print(f'\nEpoch {epochs_completed} took: {epoch_time:.3f} seconds')
            print(f'\n{(train_time/epoch_time)*100:.1f}% of the time was spent on training.\n'
                    f'{(valid_time/epoch_time)*100:.1f}% of the time was spent on validation.\n'
                    f'\nTraining/validating for: {total_epoch_time:.2f} seconds this session...')
        print(f'Estimated time left: {(total_epoch_time/(epoch+1))*(input_args.epochs-(epoch+1)):.2f} seconds...\n')
        
    # DONE: Do testing on the test set
    running_test_loss = 0
    accuracy = 0

    # set our model to evaluation mode
    model.eval()

    # execute code without gradient calculations
    with torch.no_grad():
        for inputs, labels in testloader:
            # send inputs and labels to gpu if available
            inputs, labels = inputs.to(device), labels.to(device)
        
            # get log probabilities from forward pass through network
            logps = model.forward(inputs)
            # get loss
            test_loss = criterion(logps, labels)
        
            # add the loss for this batch to the running test loss
            running_test_loss += test_loss.item()
        
            # calculate the accuracy for this batch
            # convert log probabilities to probabilities
            ps = torch.exp(logps)
            # get top probability and top class with topk
            top_p, top_class = ps.topk(1, dim=1)
            # if top class and label match, set equals to True
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    test_loss_achieved = running_test_loss/len(testloader)
    test_accuracy_achieved = accuracy/len(testloader)
        
    print(f'Test loss: {test_loss_achieved:.3f}... '
          f'Test accuracy: {test_accuracy_achieved:.3f}...')

    running_test_loss = 0
    model.train();
    
    # DONE: Save the checkpoint
    image_datasets = {'train': trainset, 'valid': validset, 'test': testset}
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {'input_size': 2208 if input_args.arch == "densenet" else 25088,
                  'output_size': 102,
                  'batch_size': batch_size,
                  'model': models.densenet161(pretrained=True) if input_args.arch == "densenet" else models.vgg19_bn(pretrained=True),
                  'classifier': model.classifier,
                  'epochs_completed': epochs_completed,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'test_loss': test_loss_achieved,
                  'test_accuracy': test_accuracy_achieved}
    
    print(f"Saving checkpoint to '{input_args.save_dir}'")
    torch.save(checkpoint, input_args.save_dir)
    

def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    the program is run from a terminal. Default arguments are parsed if the user
    does not specify them.
    Command Line Arguments:
      1. Data directory as data_dir with no default
      2. CNN Model Architecture as --arch with the default value 'densenet'
      3. Initial Learning Rate as --lr with the default value 0.01
      4. Number of Hidden Units as --neurons with the default value 4096
      5. Number of additional Hidden Layers as --layers with the default value of 0
      6. Number of training epochs as --epochs with the default value of 100
      7. Batch size as --batch_size with the default value of 64
      8. Scheduler patience as --patience with the default value of 10
      9. Scheduler lr_factor as --lr_factor with the default value of 0.5
      10. Training on GPU as --gpu if flag parsed
      11. Save directory as --save_dir with the default value of 'checkpoint.pth'
      12. Validation every x epochs as --valid_every with the default value of 1
      13. Verbose reporitng as --verbose with the default of False
    This function returns these arguments as an ArgumentParser object.
    Parameters:
      None - simply using argparse module to create and store command line arguments
    Returns:
      parse_args() - data structure that stores the command line arguments object
    """
    # create parser object with ArgumentParser
    parser = argparse.ArgumentParser()
    
    # add the command line arguments to the parser object using add_argument() method
    # data directory
    parser.add_argument('data_dir', action = "store", type = str, help = 'Base data directory that contains train, valid and test folders')
    # network
    parser.add_argument('--arch', type = str, default = 'densenet', help = 'CNN architecture to use (densenet/vgg)')
    # learning rate
    parser.add_argument('--lr', type = float, default = 0.01, help = 'Initial learning rate to use (0.01 is a good start)')
    # number of hidden units
    parser.add_argument('--neurons', type = int, default = 4096, help = 'Number of hidden units/\'neurons\' per fully-connected layer')
    # number of additional fc layers
    parser.add_argument('--layers', type = int, default = 0, help = 'Number of additional hidden layers')
    # number of training epochs
    parser.add_argument('--epochs', type = int, default = 100, help = 'Number of training epochs')
    # batch size
    parser.add_argument('--batch_size', type = int, default = 64, help = 'Batch size')
    # scheduler patience
    parser.add_argument('--patience', type = int, default = 10, help = 'Number of epochs without training loss improvement after which the learning rate is reduced')
    # scheduler lr_factor
    parser.add_argument('--lr_factor', type = float, default = 0.5, help ='Factor by which the learning rate is reduced after patience is passed')
    # train on gpu
    parser.add_argument('--gpu', action = "store_true", default = False, help = 'CUDA-capable GPU is attempted to be used if this argument is present')
    # save directory
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Save directory for checkpoint file e.g. /checkpoints/awesomeclassifier.pth')
    # validation every x epochs
    parser.add_argument('--valid_every', type = int, default = 1, help = 'Do validation pass every this many epochs.')
    # verbose reporting
    parser.add_argument('--verbose', action = "store_true", default = False, help = 'Verbose reporting if this argument is present')
    
    # return the collection of arguments
    return parser.parse_args()


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


def optimizer_first():
    """
    Checks torch version and if >=1.1.0 returns True. Fixes backward-compatibility of optim.lr_scheduler,
    i.e. optimizer.step() before scheduler.step() for versions >1.1.0.
    Parameters:
      None - simply using torch.__version__ to check torch version
    Returns:
      optimizer_first - bool, True if torch version is >=1.1.0, False if not
    """
    
    # get torch version
    torch_version = torch.__version__
    # check if version >=1.1.0 (for scheduler/optimizer.step() order)
    optimizer_first = int(torch_version[0]) >= 1 and int(torch_version[2]) >= 1
    
    return optimizer_first


def build_classifier_dict(convnet, num_hidden_layers, num_fc_neurons):
    """
    Accepts 'densenet' or 'vgg' as a valid CNN architecture, the number of additional 
    fully-connected layers and the number of neurons per layer and returns an OrderedDict 
    to be used with nn.Sequential to create the classifier of our network.
    Parameters:
      convnet - a string identifying the CNN architecture being used ('densenet' or 'vgg' only)
      num_hidden_layers - an int of the number of hidden layers to be used
      num_fc_neurons - an int of the number of neurons in each fully-connected layer
    Returns:
      classifier_dict - an OrderedDict describing the fully-connected layers, SELUs, AlphaDropout and LogSoftmax
    """
    
    classifier_dict = OrderedDict([])
    convnet_out = 1 
    
    if convnet == 'densenet':
        convnet_out = 2208
    elif convnet == 'vgg':
        convnet_out = 25088
    else:
        # this should never run due to the else: statement in main() that exits if invalid CNN arch selected
        print(f"WARNING: Network type of '{convnet}' was parsed for which the number of output neurons is unknown.\n\t Your network will not be built correctly.\n")
    
    classifier_dict['fc1'] = nn.Linear(convnet_out, num_fc_neurons)
    classifier_dict['selu1'] = nn.SELU()
    classifier_dict['dropout1'] = nn.AlphaDropout(p=0.5)
    
    for layer in range(num_hidden_layers):
        classifier_dict['fc'+str(layer+2)] = nn.Linear(num_fc_neurons, num_fc_neurons)
        classifier_dict['selu'+str(layer+2)] = nn.SELU()
        classifier_dict['dropout'+str(layer+2)] = nn.AlphaDropout(p=0.5)
    
    classifier_dict['fc'+str(num_hidden_layers+2)] = nn.Linear(num_fc_neurons, 102)
    classifier_dict['output'] = nn.LogSoftmax(dim=1)
    
    return classifier_dict


def classifier_setup(model, convnet, num_hidden_layers, num_fc_neurons, lr, epochs, patience, lr_factor, gpu, num_batches):
    """
    Returns device, criterion, optimizer and scheduler after setting up
    a classifier network to attach to our feature network.
    Parameters:
      model - model to be used
      convnet - a string identifying the CNN architecture being used
      num_hidden_layers - an int of the number of additional fully-connected layers to be used
      num_fc_neurons - an int of the number of neurons in each fully-connected layer
      lr - initial learning rate
      epochs - number of epochs to train for
      patience - patience before scheduler reduces learning rate
      lr_factor - factor by which learning rate is decreased when patience is reached
      gpu - whether gpu is used or not
      num_batches - number of batches for normalizing patience to epochs
    Returns:
      model - the model to be used
      device - the torch device returned from the use_device() function call
      criterion - the loss function specified in this function
      optimizer - the optimizer specified in this function
      scheduler - the scheduler specified in this function
    """
    
    device = use_gpu(gpu)
    
    classifier_dict = build_classifier_dict(convnet, num_hidden_layers, num_fc_neurons)
    
    # create our classifier network to bolt on behind the pretrained convnet
    classifier = nn.Sequential(classifier_dict)
    
    # change the pretrained model's classifier to our classifier
    if convnet == "densenet" or convnet == "vgg":
        model.classifier = classifier
    else:
        print(f"We shouldn't be here, Dave.\n"
              f"(Invalid feature network: {convnet} inputted to function.)\n")
        exit()
    
    # send our model to the gpu if available and selected
    model.to(device)
    
    # set our loss function to Negative Log Likelihood Loss
    criterion = nn.NLLLoss()
    
    # set our optimizer and have it optimize only the classifier parameters (not those of the convnet)
    optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9) # lr=0.0005 for Adam, 0.01 for SGD and momentum=0.9
    
    # scheduler patience
    patience = patience
    
    # when scheduler.step() comes before optimizer.step() (pre torch 1.1.0), multiply by number of batches so patience is 
    # still based on epochs
    if optimizer_first() == False:
        patience = patience * num_batches
    
    # scheduler for LR decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True, factor=lr_factor)
    
    print(f"\nFeature network classifier replaced. You specified:\n"
          f" {num_hidden_layers} additional hidden layers.\n"
          f" {num_fc_neurons} neurons per hidden layer.\n\n"
          f"Your classifier will be trained for {epochs} epochs with a learning rate of {lr}.\n")
    if optimizer_first():
        print(f"The scheduler will reduce this by a factor of {lr_factor} after {patience} epochs of\n"
              f"no reduction in training loss.")
    else:
        print(f"The scheduler will reduce this by a factor of {lr_factor} after {int(patience/num_batches)} epochs of\n"
              f"no reduction in training loss.")
    
    return model, device, criterion, optimizer, scheduler


# call main and run the program
if __name__ == "__main__":
    main()