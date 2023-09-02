import json
import glob
import numpy as np
from PIL import Image
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

unlabel_folder = os.path.join(os.getcwd(), "Unlabelled_data_preprocessing")

# with open('[YOUR JSON FILES].json') as f:
#   data_bb = json.load(f)

with open(os.path.join(unlabel_folder, "fish_counting_sample_region.json")) as f:
    data_bb = json.load(f)

videos = list(data_bb.keys())

# Optional: Slicing the list to run in low-memory devices
videos = videos[::2]

video_files = []
# obtain videoname#framenumber from videoname
for ind in videos:
    # Assumes image file named as videoname#framenumber: YYYY-MM-DD_HHMMSS#XXXX
    # video_files难道不会出现很多重复的视频吗
    video_files.append(ind[:17])

# Remove duplicate video files
video_files = list(set(video_files))

std_width = 320
std_height = 576


def get_bounding_box(i, dataset, image_list, dataset_indices):
    labels = dataset[image_list[dataset_indices[i]]]['regions']

    for l in range(0, len(labels)):
        if 'main_bb' in labels[l]['region_attributes'].values():
            label_bb = labels[l]['shape_attributes']
            label_bb = label_bb.values()
            label_bb = list(label_bb)
            break
    return label_bb


def random_translation(width, height):
    left = random.randint(0, std_width - width)
    right = left + width
    upper = random.randint(0, std_height - height)
    lower = upper + height

    return left, right, upper, lower


def three_quarters(img):
    # crop & random vertical flip
    img_crop = img.crop((0, 0.25 * std_height, 0.75 * std_width, std_height))
    flip = random.randint(0, 1)
    if flip == 1:
        img_crop = img_crop.transpose(Image.FLIP_LEFT_RIGHT)
    if flip == 0:
        img_crop = img_crop

    return img_crop


def half(img):

    # crop & random vertical flip
    img_crop = img.crop((0, 0.5 * std_height, 0.5 * std_width, std_height))
    flip = random.randint(0, 1)
    if flip == 1:
        img_crop = img_crop.transpose(Image.FLIP_LEFT_RIGHT)
    if flip == 0:
        img_crop = img_crop
    return img_crop


def one_quarter(img):

    # crop & random vertical flip
    img_crop = img.crop((0, 0.75 * std_height, 0.25 * std_width, std_height))
    flip = random.randint(0, 1)
    if flip == 1:
        img_crop = img_crop.transpose(Image.FLIP_LEFT_RIGHT)
    if flip == 0:
        img_crop = img_crop
    return img_crop


# patches = len(video_files)*1500
patches = len(videos)

count_images = 0
count_non_blank = 0

full_arr = np.zeros((patches, std_height, std_width, 3))
three_quarter_arr = np.zeros((patches, std_height, std_width, 3))
half_arr = np.zeros((patches, std_height, std_width, 3))
one_quarter_arr = np.zeros((patches, std_height, std_width, 3))

for i in range(len(video_files)):

    # path = ('/{}'.format(video_files[i]))
    path = os.path.join(os.getcwd(), "data", video_files[i])

    # 得到每个图像region字段的值，赋值给labels
    # data_bb是整个json文件，最外层属性是frame名
    # Find the corresponding bounding box for this video file
    matching_img = None
    for im in videos:
        if video_files[i] in im:
            matching_img = im
            break

    labels = data_bb[matching_img]['regions']
    # 从region的值获取shape_attributes字段，即bounding box
    # 因为region是个数组，所以labels也是一个数组，数组只有一个元素
    # 这个values就是将key-value转换为value数组
    bb = list(labels[0]['shape_attributes'].values())
    # 应该是缩放因子
    rescale = std_width / bb[3]
    # The folder "8-17m" contains images selected from the "unlabel_images" folder, which only include images with a distance range of 8-17 meters.
    img_paths = [os.path.join(unlabel_folder, "8-17m", item[:26])
                 for item in videos if item.startswith(os.path.basename(path))]
    # 这个img_paths应该存放的是，一个视频文件与之对应的所有图像frame文件

    for file in img_paths:
        img = Image.open("{}".format(file))
        count_images += 1

        # 一个图像只取它的sample_region，就是论文中8m-17m
        img1 = img.crop((bb[1], bb[2], bb[1] + bb[3], bb[2] + bb[4]))

        newsize = (int(img1.size[0] * rescale), int(img1.size[1] * rescale))
        # 裁剪完后，resize成论文中要求的320*576
        img2 = img1.resize(newsize, Image.BILINEAR)
        # img应该是确保大小是320*576
        img3 = img2.crop((0, 0, std_width, std_height))

        # 这个应该用来确保图像里是有鱼的，不是全黑的图像
        if np.sum(img3) > 1000:
            count_non_blank += 1
            print(count_non_blank)

            # full size
            # 初始化full_arr里一个图片
            data = np.asarray(img3, dtype="float32")  # check dtype
            data = data / 255

            gri_im = np.zeros((std_height, std_width, 3), dtype=np.float32)
            gri_im[0:std_height, 0:std_width, 0:3] = data

            full_arr[count_non_blank - 1] = gri_im

            # three quarter size
            gri_im = np.zeros((std_height, std_width, 3), dtype=np.float32)

            img4 = three_quarters(img3)
            data2 = np.asarray(img4, dtype="float32")  # check dtype
            data2 = data2 / 255

            width = img4.size[0]
            height = img4.size[1]
            # 将crop后的图像放到一个空白背景中，下面4个值决定了crop图像在空白背景的位置
            left, right, upper, lower = random_translation(width, height)

            gri_im[upper:lower, left:right, :] = data2

            three_quarter_arr[count_non_blank - 1] = gri_im

           # half size
            gri_im = np.zeros((std_height, std_width, 3), dtype=np.float32)

            img5 = half(img3)
            data3 = np.asarray(img5, dtype="float32")  # check dtype
            data3 = data3 / 255

            width = img5.size[0]
            height = img5.size[1]

            left, right, upper, lower = random_translation(width, height)

            gri_im[upper:lower, left:right, :] = data3

            half_arr[count_non_blank - 1] = gri_im

            # 1/4 size
            gri_im = np.zeros((std_height, std_width, 3), dtype=np.float32)

            img6 = one_quarter(img3)
            data4 = np.asarray(img6, dtype="float32")  # check dtype
            data4 = data4 / 255

            width = img6.size[0]
            height = img6.size[1]
            # 随机平移图像，因为crop后的图像要放在一个空白图像上，使得所有输入图像的size都是相等的
            left, right, upper, lower = random_translation(width, height)

            gri_im[upper:lower, left:right, :] = data4

            one_quarter_arr[count_non_blank - 1] = gri_im


print(count_non_blank)
# trim arrays to number of images augmented
full = full_arr[0:count_non_blank, :, :, :]
three_quarter = three_quarter_arr[0:count_non_blank, :, :, :]
half = half_arr[0:count_non_blank, :, :, :]
one_quarter = one_quarter_arr[0:count_non_blank, :, :, :]


# randomly chose pairs - the same number of pairs as labelled train set

num_train = 350  # change to size of labelled training data
num_val = 70
num_test = 80
num_lab_img = num_train + num_val + num_test  # number of train, val, test images

all_data = [one_quarter, half, three_quarter, full]


# choose to generate 1 pair per image or 6 pairs per image (comment / comment out as appropriate):


"""This is the code for generating one pair per original image"""

pair_large = np.zeros((num_lab_img, std_height, std_width, 3))
pair_small = np.zeros((num_lab_img, std_height, std_width, 3))

order = np.random.permutation(count_non_blank)

for i in range(num_lab_img):
    a = random.randint(1, 3)  # can set a=3 (or a = random.randint(2,3)) to increase likelihood for pair 1 > pair 2 (not equal to), if this generates sufficient number of sample pairs
    b = random.randint(0, a - 1)
    pair_large[i] = all_data[a][order[i]]
    pair_small[i] = all_data[b][order[i]]

""""""

"""This is the code for generating 6 pairs per original image"""

# pair_large_all = np.zeros((len(full)*6, std_height, std_width, 3))
# pair_small_all = np.zeros((len(full)*6, std_height, std_width, 3))

# pairs = 0
# for j in range(len(full)):
#   pair_large_all[pairs] = all_data[3][j]
#   pair_small_all[pairs] = all_data[2][j]
#   pairs += 1
# for j in range(len(full)):
#   pair_large_all[pairs] = all_data[3][j]
#   pair_small_all[pairs] = all_data[1][j]
#   pairs += 1
# for j in range(len(full)):
#   pair_large_all[pairs] = all_data[3][j]
#   pair_small_all[pairs] = all_data[0][j]
#   pairs += 1
# for j in range(len(full)):
#   pair_large_all[pairs] = all_data[2][j]
#   pair_small_all[pairs] = all_data[1][j]
#   pairs += 1
# for j in range(len(full)):
#   pair_large_all[pairs] = all_data[2][j]
#   pair_small_all[pairs] = all_data[0][j]
#   pairs += 1
# for j in range(len(full)):
#   pair_large_all[pairs] = all_data[1][j]
#   pair_small_all[pairs] = all_data[0][j]
#   pairs += 1

# # 打乱pair_large和pair_small的下标
# order = np.random.permutation(len(pair_large_all))

# pair_large = pair_large_all[order]
# pair_small = pair_small_all[order]

""""""

# 打乱完，然后再从开头开始选择和label image一样多的数据，这样就相当于随机在unlabel中选择和label数量一样多的数据
pair1_train = pair_large[0:num_train]
pair2_train = pair_small[0:num_train]
pair1_val = pair_large[num_train:num_train + num_val]
pair2_val = pair_small[num_train:num_train + num_val]
pair1_test = pair_large[num_train + num_val:num_train + num_val + num_test]
pair2_test = pair_small[num_train + num_val:num_train + num_val + num_test]

np.save('./data/unlabelled_data/pair1_train', pair1_train)
np.save('./data/unlabelled_data/pair2_train', pair2_train)
np.save('./data/unlabelled_data/pair1_val', pair1_val)
np.save('./data/unlabelled_data/pair2_val', pair2_val)
np.save('./data/unlabelled_data/pair1_test', pair1_test)
np.save('./data/unlabelled_data/pair2_test', pair2_test)
