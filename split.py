from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os, pdb
import shutil


all_images = os.listdir("raw")
all_images = [i for i in all_images if "jpg" in i]
all_images = sorted(all_images)

files = os.listdir('.')
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




cnt = 0
for class_images in train_images:
    for img in class_images:                                 
        class_i = "class_" + str(cnt)
        os.makedirs(f"flower/train/{class_i}", exist_ok=True)
        os.system(f"cp raw/{img} flower/train/{class_i}")
    print("train", class_i)
    cnt += 1
    
cnt = 0
for class_images in val_images:
    for img in class_images:                                 
        class_i = "class_" + str(cnt)
        os.makedirs(f"flower/val/{class_i}", exist_ok=True)
        os.system(f"cp raw/{img} flower/val/{class_i}")
    print("val", class_i)
    cnt += 1


cnt = 0
for class_images in test_images:
    for img in class_images:                                 
        class_i = "class_" + str(cnt)
        os.makedirs(f"flower/test/{class_i}", exist_ok=True)
        os.system(f"cp raw/{img} flower/test/{class_i}")
    print("test", class_i)
    cnt += 1
