import torch
import torchvision
import torchvision.transforms as T
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
transform0 = T.Resize(256)
transform1 = T.RandomRotation((-45, 45))
transform2_1 = T.RandomVerticalFlip(1)
transform2_2 = T.RandomHorizontalFlip(1)
transform3 = T.RandomResizedCrop(size=(224, 224))
transform4 = T.RandomAffine(translate=(0.1, 0.3))
transform5 = T.Resize(224)

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
        img1 = Image.fromarray(img, 'RGB')
        img1 = img1.resize((224, 224))
        img1.save(f"train/{class_i}/{img_dir}")

        # augmentating the image by random choice
        aug_choice = random.randint(1, 4)
        if aug_choice == 1:
            aug_image = transform1(aug_image)

        elif aug_choice == 2:
            b = randint(1,2)
            if b == 1:
                aug_image = transform2_1(aug_image)
            else:
                aug_image = transform2_2(aug_image)
                
        elif aug_choice == 3:
            aug_image = transform0(aug_image)
            aug_image = transfrom3(aug_image)
            
        elif aug_choice == 4:
            aug_image = transform5(aug_image)
            aug_image = transfrom4(aug_image)
            
        aug_image = Image.fromarray(aug_image, 'RGB')
        aug_image = aug_image.resize((224, 224))
        aug_image.save(f"train/{class_i}/{img_dir}")
        
        #aug_img = Image.fromarray(aug_img, 'RGB')
        
        #aug_img.save(f"train/{class_i}/aug_{img_no}_{aug_choice}.jpg")
    print("train", class_i)
    cnt += 1
    
    
    
cnt = 0
for class_images in val_images:
    for img in class_images:                                 
        class_i = "class_" + str(cnt)
        os.makedirs(f"val/{class_i}", exist_ok=True)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((224, 224))
        img.save(f"val/{class_i}")
    print("val", class_i)
    cnt += 1


cnt = 0
for class_images in test_images:
    for img in class_images:                                 
        class_i = "class_" + str(cnt)
        os.makedirs(f"test/{class_i}", exist_ok=True)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((224, 224))
        img.save(f"test/{class_i}")
    print("test", class_i)
    cnt += 1
