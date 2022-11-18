
# %%
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
from sklearn.metrics import confusion_matrix
from model import densenet
from torch.utils.tensorboard import SummaryWriter
import seaborn as sn
import pandas as pd
# %%
cudnn.benchmark = True
plt.ion()   # interactive mode


# %%
# Data augmentation and normalization for training
data_transforms = {
        
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),    
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ### Visualize a few images
# Let's visualize a few training images so as to understand the data
# augmentations. 
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

# ## Training the model
# %%
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, writer_name = ''):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    x_epoch = np.arange(num_epochs)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
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

            if phase == 'train' and scheduler != None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.cpu())
                writer.add_scalar("Loss/train", epoch_loss, epoch)
                writer.add_scalar("acc/train", epoch_acc, epoch)
                writer.flush()
            elif phase == 'val':
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc.cpu())
                writer.add_scalar("Loss/val", epoch_loss, epoch)
                writer.add_scalar("acc/val", epoch_acc, epoch)
                writer.flush()
            '''
            elif phase == 'test':
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc.cpu())
            '''
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        fig = plt.figure()
        ax0 = fig.add_subplot(121, title="loss")
        ax1 = fig.add_subplot(122, title="top1acc")

        ax0.plot(x_epoch[:epoch + 1], train_loss, 'b-', label = 'train')
        ax0.plot(x_epoch[:epoch + 1], val_loss, 'r-', label = 'val')
        #ax0.plot(x_epoch, test_loss, 'g-', label = 'test')
        ax1.plot(x_epoch[:epoch + 1], train_acc, 'b-', label = 'train')
        ax1.plot(x_epoch[:epoch + 1], val_acc, 'r-', label = 'val')
        #ax1.plot(x_epoch, test_acc, 'g-', label = 'test')
        
        ax0.legend()
        ax1.legend()
        
        
        print()
    fig.savefig(os.path.join('./lossGraphs', f'{writer_name}.jpg'))
    writer.add_figure(tag = 'Graphs', figure= fig)
    writer.flush()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# ### Visualizing the model predictions
# Generic function to display predictions for a few images

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

def test_model(model, criterion):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    
    y_pred = []
    y_true = []

        
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            y_true.extend(labels.data.cpu().numpy())
            labels = labels.to(device)
            

            # forward
            # track history if only in train
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.data.cpu().numpy())
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        test_loss = running_loss / dataset_sizes['test']
        test_acc = running_corrects.double() / dataset_sizes['test']
        cf_matrix = confusion_matrix(y_true, y_pred, normalize= 'true')

        df_cm = pd.DataFrame(cf_matrix, index = [i for i in class_names],
                     columns = [i for i in class_names])
        
        
        fig = plt.figure(figsize = (20,12))
        sns = sn.heatmap(df_cm, annot=True)
        fig = sns.get_figure()
        fig.savefig(f'Confusion_Matrix/{writer_name}.png')
        writer.add_figure(tag = 'Confusion Matrix', figure= fig)
        writer.flush()
        
    print(f'test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    # deep copy the model
            

    time_elapsed = time.time() - since


    # load best model weights
    model.load_state_dict(best_model_wts)
    return test_loss, test_acc


# %%
model_name = 'densenet121'
optimizer_name = 'adam'
number_of_epoch = 50
batch_size = 4
lr = 1e-4
L2_lambda = 1e-5

#model = vgg.vgg16(pretrained=False)

# Data loading
data_dir = 'flower'


model = getattr(densenet, model_name)()

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                for x in ['train', 'val','test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                            shuffle=True, num_workers=4)
            for x in ['train', 'val','test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

model_ft = model.to(device)

criterion = nn.CrossEntropyLoss()


gamma = 0.1
momentum = 0.9

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.Adam(model.parameters(), lr= lr)
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum= momentum)

# Decay LR by a factor of 0.1 every 7 epochs
step_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=13, gamma=gamma)


# ### Train and evaluate
# %%
writer_name = f"{model_name}_{optimizer_name}_lr_{lr}_gamma_{gamma}_batch_size_{batch_size}"

writer = SummaryWriter(comment = writer_name)
model_ft = train_model(model_ft, criterion, optimizer_ft, step_lr_scheduler,
                    num_epochs=number_of_epoch, writer_name = writer_name)

# %%


test_loss, test_acc = test_model(model, criterion)

writer.add_hparams({
        "batch_size": batch_size,
        "learning_rate": lr,
        "model_name": f"{model_name}",
        "optimizer": optimizer_name,
        "num_of_epoch": number_of_epoch
        },
        {
        "test_acc": test_acc,
        "test_loss" : test_loss}
        )
        
writer.flush()
writer.close()






# %%
