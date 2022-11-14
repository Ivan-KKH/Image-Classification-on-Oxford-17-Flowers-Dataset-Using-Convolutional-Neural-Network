# %%
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np
import os, pdb
import random
from PIL import Image


all_files = os.listdir("train/.")

all_classes = [i for i in all_files if "class" in i]

for classes in all_classes:
    files_in_class = os.listdir(f'train/{classes}/')
    all_image_files = [i for i in files_in_class if "image" in i]
    

    #os.makedirs(f"train/{classes}/preview", exist_ok=True)
    
    dataset = []
    for img in all_image_files:
        x = io.imread(f'train/{classes}/{img}')
        x = Image.fromarray(x, 'RGB')
        x = x.resize((500,500))
        dataset.append(np.array(x))
    all_images = np.array(dataset)
    all_image_files = [i.replace("image_", "") for i in all_image_files]
    all_image_files = [i.replace(".jpg", "") for i in all_image_files]
    

    datagen0 = ImageDataGenerator()
    datagen1 = ImageDataGenerator(rotation_range=30, fill_mode='nearest')
    datagen2 = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)
    datagen3 = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    datagen4 = ImageDataGenerator(zoom_range=0.3)

    '''
    # datagen0 = Unaugmented data
    j=0
    for image in all_images:
        image = image.reshape((1, ) + image.shape) 
        
        i = 0

        for batch in datagen0.flow(image, batch_size=16, save_to_dir=f'train/{classes}/preview', save_prefix='aug' + all_image_files[j], save_format='jpg'):
            i += 1
            j += 1
            break
    '''
    j = 0
    for image in all_images:
        image = image.reshape((1, ) + image.shape) 
        a = random.randint(1, 4)
        if a == 1:
            i = 0
            for batch in datagen1.flow(image, batch_size=16, save_to_dir=f'train/{classes}/preview', save_prefix='aug' + '_' + all_image_files[j] + '_' + str(a), save_format='jpg'):
                i += 1
                j += 1
                break
        if a == 2:
            i = 0
            for batch in datagen2.flow(image, batch_size=16, save_to_dir=f'train/{classes}/preview', save_prefix='aug' + '_' + all_image_files[j] + '_' + str(a), save_format='jpg'):
                i += 1
                j += 1
                break
        if a == 3:
            i = 0
            for batch in datagen3.flow(image, batch_size=16, save_to_dir=f'train/{classes}/preview', save_prefix='aug' + '_' + all_image_files[j] + '_' + str(a), save_format='jpg'):
                i += 1
                j += 1
                break
        if a == 4:
            i = 0
            for batch in datagen4.flow(image, batch_size=16, save_to_dir=f'train/{classes}/preview', save_prefix='aug' + '_' + all_image_files[j] + '_' + str(a), save_format='jpg'):
                i += 1
                j += 1
                break

    # %%
