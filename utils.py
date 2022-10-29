from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
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


def create_confusion_matrix(y_pred:list, y_true:list):
    # Build confusion matrix
    with open('flower/class_name.txt') as file:
        classes = [line.rstrip() for line in file]
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        cf_matrix/cf_matrix.astype(np.float64).sum(axis=1), 
        index=[i for i in classes],
        columns=[i for i in classes]
    )
    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt=".2f").get_figure()


# ## Training the model
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
        fig.savefig(os.path.join('./lossGraphs', f'train_VGG.jpg'))
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model