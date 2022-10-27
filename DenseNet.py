# %%
%matplotlib inline

# %% [markdown]
# 
# # Transfer Learning for Computer Vision Tutorial
# **Author**: [Sasank Chilamkurthy](https://chsasank.github.io)
# 
# In this tutorial, you will learn how to train a convolutional neural network for
# image classification using transfer learning. You can read more about the transfer
# learning at [cs231n notes](https://cs231n.github.io/transfer-learning/)_
# 
# Quoting these notes,
# 
#     In practice, very few people train an entire Convolutional Network
#     from scratch (with random initialization), because it is relatively
#     rare to have a dataset of sufficient size. Instead, it is common to
#     pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
#     contains 1.2 million images with 1000 categories), and then use the
#     ConvNet either as an initialization or a fixed feature extractor for
#     the task of interest.
# 
# These two major transfer learning scenarios look as follows:
# 
# -  **Finetuning the convnet**: Instead of random initialization, we
#    initialize the network with a pretrained network, like the one that is
#    trained on imagenet 1000 dataset. Rest of the training looks as
#    usual.
# -  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
#    for all of the network except that of the final fully connected
#    layer. This last fully connected layer is replaced with a new one
#    with random weights and only this layer is trained.
# 

# %%
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()   # interactive mode

# %% [markdown]
# ## Load Data
# 
# We will use torchvision and torch.utils.data packages for loading the
# data.
# 
# The problem we're going to solve today is to train a model to classify
# **ants** and **bees**. We have about 120 training images each for ants and bees.
# There are 75 validation images for each class. Usually, this is a very
# small dataset to generalize upon, if trained from scratch. Since we
# are using transfer learning, we should be able to generalize reasonably
# well.
# 
# This dataset is a very small subset of imagenet.
# 
# .. Note ::
#    Download the data from
#    [here](https://download.pytorch.org/tutorial/hymenoptera_data.zip)
#    and extract it to the current directory.
# 
# 
'''
transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
'''
# %%
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    ]),
    
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),    
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'flower'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ### Visualize a few images
# Let's visualize a few training images so as to understand the data
# augmentations.
# 
# 

# %%
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

# %% [markdown]
# ## Training the model
# 
# Now, let's write a general function to train a model. Here, we will
# illustrate:
# 
# -  Scheduling the learning rate
# -  Saving the best model
# 
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.
# 
# 

# %%

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss = []
    val_loss = []
    test_loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    x_epoch = np.arange(num_epochs)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.cpu())
            elif phase == 'val':
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc.cpu())
            elif phase == 'test':
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc.cpu())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1acc")

    ax0.plot(x_epoch, train_loss, 'bo-', label = 'train')
    ax0.plot(x_epoch, val_loss, 'ro-', label = 'val')
    ax0.plot(x_epoch, test_loss, 'go-', label = 'test')
    ax1.plot(x_epoch, train_acc, 'bo-', label = 'train')
    ax1.plot(x_epoch, val_acc, 'ro-', label = 'val')
    ax1.plot(x_epoch, test_acc, 'go-', label = 'test')
    
    ax0.legend()
    ax1.legend()
    fig.savefig(os.path.join('./lossGraphs', f'train_DenseNet.jpg'))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# %% [markdown]
# ### Visualizing the model predictions
# 
# Generic function to display predictions for a few images
# 
# 
# 

# %%
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# %% [markdown]
# ## Finetuning the convnet
# 
# Load a pretrained model and reset final fully connected layer.
# 
# 
# 


# %%
model_ft = models.densenet121(pretrained=False)
num_ftrs = model_ft.classifier.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.classifier = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# %%


# %% [markdown]
# ### Train and evaluate
# 
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
# 
# 
# 

# %%
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)


# %%
visualize_model(model_ft)

# %% [markdown]
# ## ConvNet as fixed feature extractor
# 
# Here, we need to freeze all the network except the final layer. We need
# to set ``requires_grad = False`` to freeze the parameters so that the
# gradients are not computed in ``backward()``.
# 
# You can read more about this in the documentation
# [here](https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward)_.
# 
# 
# 
