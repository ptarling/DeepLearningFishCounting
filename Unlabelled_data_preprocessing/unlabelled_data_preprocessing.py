
import json
import glob
import numpy as np
from PIL import Image
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"





with open('[YOUR JSON FILES].json') as f:
  data_bb = json.load(f)


videos = list(data_bb.keys())


video_files = []
for ind in videos:
            video_files.append(ind[:17]) #Assumes image file named as videoname#framenumber: YYYY-MM-DD_HHMMSS#XXXX
       
    
std_width = 320
std_height = 576
        

def get_bounding_box(i, dataset, image_list, dataset_indices):
    labels = dataset[image_list[dataset_indices[i]]]['regions']
    
    for l in range(0,len(labels)):
            if 'main_bb' in labels[l]['region_attributes'].values():
                label_bb = labels[l]['shape_attributes']
                label_bb = label_bb.values()
                label_bb = list(label_bb)
                break
    return label_bb


def random_translation(width, height):
    left = random.randint(0,std_width-width)
    right = left + width
    upper = random.randint(0,std_height-height)
    lower = upper + height
    
    return left, right, upper, lower



def three_quarters(img):
    #crop & random vertical flip
    img_crop = img.crop((0,0.25*std_height,0.75*std_width,std_height))
    flip = random.randint(0,1)
    if flip == 1:
        img_crop = img_crop.transpose(Image.FLIP_LEFT_RIGHT)
    if flip == 0:
        img_crop = img_crop 

    return img_crop

def half(img):
    
    #crop & random vertical flip
    img_crop = img.crop((0,0.5*std_height,0.5*std_width,std_height))
    flip = random.randint(0,1)
    if flip == 1:
        img_crop = img_crop.transpose(Image.FLIP_LEFT_RIGHT)
    if flip == 0:
        img_crop = img_crop  
    return img_crop

def one_quarter(img):
    
    #crop & random vertical flip
    img_crop = img.crop((0,0.75*std_height,0.25*std_width,std_height))
    flip = random.randint(0,1)
    if flip == 1:
        img_crop = img_crop.transpose(Image.FLIP_LEFT_RIGHT)
    if flip == 0:
        img_crop = img_crop  
    return img_crop

patches = len(video_files)*1500

count_images = 0
count_non_blank = 0

full_arr= np.zeros((patches,std_height,std_width,3))
three_quarter_arr = np.zeros((patches,std_height,std_width,3))
half_arr = np.zeros((patches,std_height,std_width,3))
one_quarter_arr = np.zeros((patches,std_height,std_width,3)) 

for i in range(len(video_files)):
    
    path = ('./{}'.format(video_files[i]))

    labels = data_bb[videos[i]]['regions']
    
    bb = list(labels[0]['shape_attributes'].values())
    
    rescale = std_width/bb[3]

    img_paths = []
    
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)


    for file in img_paths:
        img = Image.open("{}".format(file))
        count_images += 1

        img1 = img.crop((bb[1], bb[2], bb[1]+bb[3], bb[2]+bb[4]))

        newsize = (int(img1.size[0]*rescale), int(img1.size[1]*rescale))
        img2 = img1.resize(newsize, Image.BILINEAR)
        img3 = img2.crop((0, 0, std_width,std_height))

        if np.sum(img3) > 1000:
            count_non_blank +=1


            #full size
            data = np.asarray(img3, dtype="float32" ) #check dtype
            data = data / 255

            gri_im = np.zeros((std_height,std_width,3), dtype=np.float32)
            gri_im[0:std_height,0:std_width,0:3] = data

            full_arr[count_non_blank-1] = gri_im


            #three quarter size
            gri_im = np.zeros((std_height,std_width,3), dtype=np.float32)

            img4 = three_quarters(img3)
            data2 = np.asarray(img4, dtype="float32" ) #check dtype
            data2 = data2 / 255

            width = img4.size[0]
            height = img4.size[1]

            left, right, upper, lower = random_translation(width, height)

            gri_im[upper:lower,left:right,:] = data2

            three_quarter_arr[count_non_blank-1] = gri_im



           #half size
            gri_im = np.zeros((std_height,std_width,3), dtype=np.float32)

            img5 = half(img3)
            data3 = np.asarray(img5, dtype="float32" ) #check dtype
            data3 = data3 / 255

            width = img5.size[0]
            height = img5.size[1]

            left, right, upper, lower = random_translation(width, height)

            gri_im[upper:lower,left:right,:] = data3

            half_arr[count_non_blank-1] = gri_im    


            #1/4 size
            gri_im = np.zeros((std_height,std_width,3), dtype=np.float32)

            img6 = one_quarter(img3)
            data4 = np.asarray(img6, dtype="float32" ) #check dtype
            data4 = data4 / 255

            width = img6.size[0]
            height = img6.size[1]

            left, right, upper, lower = random_translation(width, height)

            gri_im[upper:lower,left:right,:] = data4

            one_quarter_arr[count_non_blank-1] = gri_im
            

#trim arrays to number of images augmented
full = full_arr[0:count_non_blank,:,:,:]
three_quarter = three_quarter_arr[0:count_non_blank,:,:,:]
half = half_arr[0:count_non_blank,:,:,:]
one_quarter = one_quarter_arr[0:count_non_blank,:,:,:]


#randomly chose pairs - the same number of pairs as labelled train set

num_train = 350 #change to size of labelled training data
num_val = 70
num_test = 80
num_lab_img = num_train + num_val + num_test #number of train, val, test images

all_data = [one_quarter, half, three_quarter, full]


#choose to generate 1 pair per image or 6 pairs per image (comment / comment out as appropriate):


"""This is the code for generating one pair per original image"""

#pair_large = np.zeros((num_lab_img,std_height,std_width,3))
#pair_small = np.zeros((num_lab_img,std_height,std_width,3))

#order = np.random.permutation(count_non_blank)

#for i in range(num_lab_img):
    #a = random.randint(1,3) #can set a=3 (or a = random.randint(2,3)) to increase likelihood for pair 1 > pair 2 (not equal to), if this generates sufficient number of sample pairs
    #b = random.randint(0,a-1) 
    #pair_large[i] = all_data[a][order[i]]
    #pair_small[i] = all_data[b][order[i]]
    
""""""

"""This is the code for generating 6 pairs per original image"""

pair_large_all = np.zeros((len(full)*6,std_height,std_width,3))
pair_small_all = np.zeros((len(full)*6,std_height,std_width,3))

pairs = 0
for j in range(len(full)):
        pair_large_all[pairs] = all_data[3][j]
        pair_small_all[pairs] = all_data[2][j]
        pairs += 1
for j in range(len(full)):
        pair_large_all[pairs] = all_data[3][j]
        pair_small_all[pairs] = all_data[1][j]
        pairs += 1
for j in range(len(full)):
        pair_large_all[pairs] = all_data[3][j]
        pair_small_all[pairs] = all_data[0][j]
        pairs += 1
for j in range(len(full)):
        pair_large_all[pairs] = all_data[2][j]
        pair_small_all[pairs] = all_data[1][j]
        pairs += 1
for j in range(len(full)):
        pair_large_all[pairs] = all_data[2][j]
        pair_small_all[pairs] = all_data[0][j]
        pairs += 1
for j in range(len(full)):
        pair_large_all[pairs] = all_data[1][j]
        pair_small_all[pairs] = all_data[0][j]
        pairs += 1
        
order = np.random.permutation(len(pair_large_all))

pair_large = pair_large_all[order]
pair_small = pair_small_all[order]

""""""
  

pair1_train = pair_large[0:num_train]
pair2_train = pair_small[0:num_train]
pair1_val = pair_large[num_train:num_train+num_val]
pair2_val = pair_small[num_train:num_train+num_val]
pair1_test = pair_large[num_train+num_val:]
pair2_test = pair_small[num_train+num_val:]

np.save('./pair1_train_new', pair1_train)
np.save('./pair2_train_new', pair2_train)
np.save('./pair1_val_new', pair1_val)
np.save('./pair2_val_new', pair2_val)
np.save('./pair1_test_new', pair1_test)
np.save('./pair2_test_new', pair2_test)
