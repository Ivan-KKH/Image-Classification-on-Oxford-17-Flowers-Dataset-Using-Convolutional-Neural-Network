# %%
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np
import os, pdb
import random
import shutil
from PIL import Image


files = os.listdir(".")

# load images
all_images = [i for i in files if "jpg" in i]
all_images = sorted(all_images)

# create train, val, test directories
if 'train' in files:
    shutil.rmtree('train')
if 'val' in files:
    shutil.rmtree('val')
if 'test' in files:
    shutil.rmtree('test')

os.makedirs("train", exist_ok=True)
os.makedirs("val", exist_ok=True)
os.makedirs("test", exist_ok= True)

# split the images into classes
train_images = []
val_images = []
test_images = []
for i in range(17):
    permutated_images = np.random.permutation(all_images[i*80 : (i+1)*80])
    train_images.append(permutated_images[:int(len(permutated_images) * 0.8)])
    val_images.append(permutated_images[int(len(permutated_images) * 0.8): int(len(permutated_images) * 0.9)])
    test_images.append(permutated_images[int(len(permutated_images) * 0.9):])

# Creating ImageGenerator
datagen1 = ImageDataGenerator(rotation_range=30, fill_mode='nearest')
datagen2 = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)
datagen3 = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
datagen4 = ImageDataGenerator(zoom_range=0.3)


# creating train set
cnt = 0
for class_images in train_images:
    for img_dir in class_images:
        #print(f"{img_dir}") 
        #get the image number
        img_no =  img_dir.replace("image_", "")
        img_no =  img_dir.replace(".jpg", "")                               

        class_i = "class_" + str(cnt)
        os.makedirs(f"train/{class_i}", exist_ok=True)
        img = io.imread(f'{img_dir}')    
        img = Image.fromarray(img, 'RGB')
        img = img.resize((224,224))
        img.save(f"train/{class_i}/{img_dir}")

        # augmentating the image by random choice
        aug_choice = random.randint(1, 4)
        aug_img = np.array(img).reshape((1, ) + np.array(img).shape) 
        if aug_choice == 1:
            for batch in datagen1.flow(aug_img, save_to_dir=f'train/{class_i}', save_prefix='aug' + '_' + img_no + '_' + str(aug_choice), save_format='jpg'):
                break
        elif aug_choice == 2:
            for batch in datagen2.flow(aug_img, save_to_dir=f'train/{class_i}', save_prefix='aug' + '_' + img_no + '_' + str(aug_choice), save_format='jpg'):
                break
        elif aug_choice == 3:
            for batch in datagen3.flow(aug_img, save_to_dir=f'train/{class_i}', save_prefix='aug' + '_' + img_no + '_' + str(aug_choice), save_format='jpg'):
                break
        elif aug_choice == 4:
            for batch in datagen4.flow(aug_img, save_to_dir=f'train/{class_i}', save_prefix='aug' + '_' + img_no + '_' + str(aug_choice), save_format='jpg'):
                break
        
        #aug_img = Image.fromarray(aug_img, 'RGB')
        
        #aug_img.save(f"train/{class_i}/aug_{img_no}_{aug_choice}.jpg")
    print("train", class_i)
    cnt += 1
    
    
    
cnt = 0
for class_images in val_images:
    for img in class_images:                                 
        class_i = "class_" + str(cnt)
        os.makedirs(f"val/{class_i}", exist_ok=True)
        os.system(f"cp {img} val/{class_i}")
    print("val", class_i)
    cnt += 1


cnt = 0
for class_images in test_images:
    for img in class_images:                                 
        class_i = "class_" + str(cnt)
        os.makedirs(f"test/{class_i}", exist_ok=True)
        os.system(f"cp {img} test/{class_i}")
    print("test", class_i)
    cnt += 1

    #os.makedirs(f"train/{classes}/preview", exist_ok=True)
'''
    dataset = []
    for img in all_image_files:
        #x = io.imread(f'train/{classes}/{img}')
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

    
    # datagen0 = Unaugmented data
    j=0
    for image in all_images:
        image = image.reshape((1, ) + image.shape) 
        
        i = 0

        for batch in datagen0.flow(image, batch_size=16, save_to_dir=f'train/{classes}/preview', save_prefix='aug' + all_image_files[j], save_format='jpg'):
            i += 1
            j += 1
            break
    
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
'''
    # %%
