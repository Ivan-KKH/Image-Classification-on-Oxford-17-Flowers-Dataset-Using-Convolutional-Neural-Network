from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import torchvision.transforms as T
from skimage import io
import os, pdb
import shutil
from PIL import Image

with open('flower/class_name.txt') as file:
    class_names = [line.rstrip() for line in file]

all_images = os.listdir("raw")
all_images = [i for i in all_images if "jpg" in i]
all_images = sorted(all_images)

files = os.listdir('flower')
if 'train' in files:
    shutil.rmtree('flower/train')
if 'val' in files:
    shutil.rmtree('flower/val')
if 'test' in files:
    shutil.rmtree('flower/test')

os.makedirs("flower/train", exist_ok=True)
os.makedirs("flower/val", exist_ok=True)
os.makedirs("flower/test", exist_ok= True)
# pdb.set_trace()

# there are 1360 images totally with 17 classes, each class has 80 images
# 1-80 belongs to class 1; 81-160 belongs to class 2...
train_images = []
val_images = []
test_images = []
for i in range(17):
    permutated_images = np.random.permutation(all_images[i*80 : (i+1)*80])
    train_images.append(permutated_images[:int(len(permutated_images) * 0.8)])
    val_images.append(permutated_images[int(len(permutated_images) * 0.8): int(len(permutated_images) * 0.9)])
    test_images.append(permutated_images[int(len(permutated_images) * 0.9):])
#print(sorted(train_images[-1]))
#print(sorted(val_images[-1]))
#print(sorted(test_images[-1]))

transform1 = T.RandomRotation((-45, 45))
transform2_1 = T.RandomVerticalFlip(1)
transform2_2 = T.RandomHorizontalFlip(1)
transform3 = T.RandomResizedCrop(size=(224, 224))
transform4 = T.RandomAffine(translate=(0.1, 0.3), degrees = 0)



cnt = 0
for class_images in train_images:
    for img in class_images: 
        img_no =  img.replace("image_", "")
        img_no =  img.replace(".jpg", "")                                 
        class_i = "class_" + str(cnt)
        class_i = class_names[cnt]
        os.makedirs(f"flower/train/{class_i}", exist_ok=True)
        image = io.imread(f'raw/{img}')
        image = Image.fromarray(image, 'RGB')
        image1 = image.resize((224,224))
        image1.save(f"flower/train/{class_i}/{img}")
        aug_choice = np.random.randint(1, 5)
        if aug_choice == 1:
            aug_image = transform1(image)

        elif aug_choice == 2:
            b = np.random.randint(1,2)
            if b == 1:
                aug_image = transform2_1(image)
            else:
                aug_image = transform2_2(image)
                
        elif aug_choice == 3:
            aug_image = transform3(image)
            
        elif aug_choice == 4:
            aug_image = transform4(image)
        aug_image = aug_image.resize((224,224))
        aug_image.save(f"flower/train/{class_i}/aug_{aug_choice}_{img}")
    
    print("train", class_i)
    cnt += 1
    
cnt = 0
for class_images in val_images:
    for img in class_images:                                 
        class_i = "class_" + str(cnt)
        class_i = class_names[cnt]
        os.makedirs(f"flower/val/{class_i}", exist_ok=True)
        image = io.imread(f'raw/{img}')
        image = Image.fromarray(image, 'RGB')
        image = image.resize((224,224))
        image.save(f"flower/val/{class_i}/{img}")
    print("val", class_i)
    cnt += 1


cnt = 0
for class_images in test_images:
    for img in class_images:                                 
        class_i = "class_" + str(cnt)
        class_i = class_names[cnt]
        os.makedirs(f"flower/test/{class_i}", exist_ok=True)
        image = io.imread(f'raw/{img}')
        image = Image.fromarray(image, 'RGB')
        image = image.resize((224,224))
        image.save(f"flower/test/{class_i}/{img}")
    print("test", class_i)
    cnt += 1
